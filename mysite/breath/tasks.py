from __future__ import absolute_import, unicode_literals
import pandas as pd
from io import BytesIO, StringIO

from collections import defaultdict, OrderedDict
from itertools import chain

from celery import shared_task, group, chord
from celery import chain as celery_chain
# from celery_progress.backend import ProgressRecorder

from celery.contrib import rdb

from django.core.files.base import ContentFile

from .celerytasks import app

from django.core.files.base import ContentFile
from django.core.files.images import ImageFile

from .models import (RawFile, ProcessedFile, WebPeakDetectionResult, MccImsAnalysisWrapper, FileSet, UnlinkedWebPeakDetectionResult, WebCustomSet, CustomPeakDetectionFileSet,
                     IntensityPlotModel, ClusterPlotModel, OverlayPlotModel, BestFeaturesOverlayPlot, ClasswiseHeatMapPlotModel,
                     DecisionTreePlotModel,
                     StatisticsModel, FeatureMatrix,
                     GCMSRawMeasurement,
                     GCMSFileSet, GCMSUnlinkedPeakDetectionResult, GCMSPeakDetectionFileSet)

from breathpy.model.BreathCore import GCMSAnalysis
from decimal import Decimal
import time
import datetime
from jobtastic import JobtasticTask

class PredictClassTask(JobtasticTask):
    # this Task uses a prediction model and predicts the classes of raw measurements
    # it needs to apply the same preprocessing options as supplied in the prediction model
    # needs to apply the same peak detection layer as selected in the model
    # The prediction model requires a Float / BitPeakAlignmentResult with the same peak_ids -
    #   so we need to reconstruct / use the same columns from model training

    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1


    def calculate_result(self, analysis_id, prediction_file_set_id, **kwargs):
        from .models import ClassPredictionFileSet, WebPredictionModel, PredictionResult
        from breathpy.model.BreathCore import ExternalPeakDetectionMethod, PredictionModel
        from shutil import rmtree
        from pathlib import Path
        import joblib
        total_progress = 5

        # get evaluation params from analysis
        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        web_prediction_model = WebPredictionModel.objects.get(mcc_ims_analysis=analysis_id)

        # us preprocess measurement method
        # give peak detection results to prediction

        classPredictionFileSet = ClassPredictionFileSet.objects.get(pk=prediction_file_set_id)
        self.update_progress(1,total_progress)

        # create tmpdir for peax if necessary
        if ExternalPeakDetectionMethod.PEAX.name in wrapper.preprocessing_options:
            tmp = wrapper.preprocessing_options[ExternalPeakDetectionMethod.PEAX.name]['tempdir']
            Path(tmp).mkdir(parents=True, exist_ok=True)

        # we created database objects for raw measurements
        raw_measurement_pks, raw_measurements = wrapper._unpack_raw_files_from_zip_prediction(prediction_file_set_id)
        # wrapper._unpack_raw_files_from_zip
        # print(raw_measurement_pks)
        preprocessing_options = wrapper.preprocessing_options

        # use chain to preprocess - first we need all peak detection results, then we can start plotting
        raw_id_processed_id_pdr_id_tuples = []
        for i, raw_file_id in enumerate(raw_measurement_pks):
            # print(raw_file_id)
            raw_id_processed_id_pdr_id_tuples.append(preprocess_measurement(raw_file_id, preprocessing_options))
            # self.update_progress(i + 1, total_progress)

        self.update_progress(3, total_progress)

        # [r_id, pr_id, [wpdr_id,], ... ]
        unzipped = list(zip(*raw_id_processed_id_pdr_id_tuples))
        print(unzipped)
        raw_ids = unzipped[0]
        processed_ids = unzipped[1]
        pdr_ids = list(chain.from_iterable(unzipped[2]))

        archive_name = classPredictionFileSet.upload.path.split("/")[-1]
        # create a Fileset for the measurements
        fileset = FileSet(name=archive_name[:50], raw_files=raw_ids, processed_files=processed_ids,
                          peak_detection_results=pdr_ids)
        fileset.save()
        fileset_id = fileset.pk

        print("Created Fileset {}".format(fileset.pk))
        # cleanup temporary files of peax directory
        if ExternalPeakDetectionMethod.PEAX.name in wrapper.preprocessing_options:
            tmp = wrapper.preprocessing_options[ExternalPeakDetectionMethod.PEAX.name]['tempdir']
            rmtree(tmp)

        mcc_ims_analysis, plot_parameters, file_parameters = wrapper.reinitialize_mcc_ims_analysis(fileset_id=fileset_id)
        mcc_ims_analysis.align_peaks(file_prefix="")

        df_matching_training = PredictionModel.reconstruct_remove_features(
            mcc_ims_analysis.peak_alignment_result.dict_of_df, web_prediction_model.feature_names_by_pdm)

        self.update_progress(4,total_progress)

        scipy_predictor_by_pdm = joblib.load(web_prediction_model.scipy_predictor_pickle.path)

        # get original labels

        original_labels = classPredictionFileSet.filename_class_label_dict
        # print(f"original_labels: {original_labels}")

        # get labels to assign class and return prediction
        # get evaluation matrix for each chosen peak_detection method
        prediction_by_pdm = dict()
        for pdm in mcc_ims_analysis.peak_detection_combined:
            X_reduced = df_matching_training[pdm.name]
            # Import exisitng model and predict classes of unknown data
            # probas = self.scipy_predictor.predict_proba(X)

            prediction_matrix = X_reduced.fillna(0)

            buff = StringIO()
            prediction_matrix.to_csv(buff, index=True, header=True, index_label="index")
            buff.seek(0)
            pm_fn = f"prediction_matrix_{pdm.name}.csv"

            # create FeatureMatrix and save in backend for potential export
            prediction_matrix_model = FeatureMatrix(
                    analysis=wrapper, name=f"Prediction Matrix {pdm.name}",
                    peak_detection_method_name=pdm.name,
                    file=ContentFile(buff.getvalue(), name=pm_fn),
                    is_training_matrix=False, is_prediction_matrix=True)
            prediction_matrix_model.save()

            prediction_by_pdm[pdm] = scipy_predictor_by_pdm[pdm].predict(prediction_matrix).tolist()
        # print(f"prediction_by_pdm[pdm] {prediction_by_pdm[pdm]}")

        labeled_prediction_result = {}

        for pdm, prediction_result in prediction_by_pdm.items():
            labeled_prediction_result[pdm] = {m.filename: web_prediction_model.class_labels[class_index] for
                                              m, class_index in zip(raw_measurements, prediction_result)}
            # print(f"labeled_prediction_result[pdm] {labeled_prediction_result[pdm]}")
            # print(f"prediction_result {prediction_result}")

        for pdm, prediction_result in labeled_prediction_result.items():
            prediction_result_model = PredictionResult(
                web_prediction_model=web_prediction_model,
                class_assignment=prediction_result,
                peak_detection_method_name=pdm.name,
                original_class_labels=original_labels,
            )
            prediction_result_model.save()

        self.update_progress(total_progress, total_progress)
        return web_prediction_model.pk



class EvaluatePerformanceTask(JobtasticTask):
    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1

    def calculate_result(self, analysis_id,  **kwargs):
        from collections import Counter
        from .models import WebPredictionModel, BoxPlotModel, RocPlotModel
        from breathpy.model.BreathCore import FeatureReductionMethod, PerformanceMeasure
        from breathpy.view.BreathVisualizations import RocCurvePlot, BoxPlot, TreePlot, ClusterPlot
        import numpy as np
        # get evaluation params from analysis
        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)
        # wrapper.ev wrapper.evaluation_options
        total_progress = 5

        self.update_progress(1, 100)

        # Create BoxPlots, decision trees and export statistics csv
        # Reinitialize mmc_ims_analysis object from DB to resume analysis
        # get preprocessed files from DB
        # evaluation parameters need to be set before reinitializing ims analysis

        mcc_ims_analysis, plot_params, file_params = wrapper.reinitialize_mcc_ims_analysis()

        # start analysis
        mcc_ims_analysis.align_peaks(file_prefix="")
        # before_reduction = mcc_ims_analysis.peak_alignment_result.dict_of_df.copy()
        mcc_ims_analysis.reduce_features([FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES])
        # after_reduction = mcc_ims_analysis.peak_alignment_result.dict_of_df.copy()

        # get number of measurements from smallest class - need at least 2 measurements for cross validation
        class_counts = Counter([m.class_label for m in mcc_ims_analysis.measurements])
        min_occurence = sorted(class_counts.values())[0]
        can_cross_validate = min_occurence >= 10

        # check if cross validation happens in BreathCore
        #
        performance_measures = []
        for eo in wrapper.evaluation_options.keys():
            try:
                # we also have feature reduction options in evaluation_options
                performance_measures.append(PerformanceMeasure(eo))
            except ValueError:
                pass

        self.update_progress(1, total_progress)

        # print(self.evaluation_options)
        # mcc_ims_analysis.evaluate_performance(performance_measures=performance_measures)
        mcc_ims_analysis.evaluate_performance()

        # export feature matrices
        for pdm_name, trainings_matrix in mcc_ims_analysis.analysis_result.trainings_matrix.items():
            fm_fn = "{}_feature_matrix".format(pdm_name)
            buff = StringIO()
            trainings_matrix.to_csv(buff, index=True, header=True, index_label="index")
            buff.seek(0)
            fm = FeatureMatrix(analysis=wrapper,
                               name=fm_fn,
                               peak_detection_method_name=pdm_name,
                               file=ContentFile(buff.getvalue(), name=fm_fn + ".csv"))
            fm.save()

        if can_cross_validate:
            # roc plots can only be done if we applied cross validation
            if len(set(mcc_ims_analysis.analysis_result.class_labels)) == 2:
                roc_plot_tuples = RocCurvePlot.ROCCurve(mcc_ims_analysis.analysis_result, plot_parameters=plot_params)
            else:
                roc_plot_tuples = RocCurvePlot.MultiClassROCCurve(mcc_ims_analysis.analysis_result,
                                                                  plot_parameters=plot_params)

            for evaluation_method_name, peak_detection_method_name, roc_plot_buffer, fig_name in roc_plot_tuples:
                roc_png = ImageFile(roc_plot_buffer, name=fig_name)
                plotModel = RocPlotModel(analysis=wrapper, user=wrapper.user,
                                         name="Roc {}".format(peak_detection_method_name),
                                         based_on_peak_detection_method_name=peak_detection_method_name,
                                         based_on_performance_measure_name=evaluation_method_name,
                                         figure=roc_png)
                plotModel.save()

        # boxplots are only created if we have corrected pvalues
        box_plot_tuples = BoxPlot.BoxPlotBestFeature(mcc_ims_analysis.analysis_result, plot_parameters=plot_params)
        for (performance_measure_method_name, peak_detection_method_name, model_of_class,
             peak_id), box_plot_buffer, fig_name in box_plot_tuples:
            box_png = ImageFile(box_plot_buffer, name=fig_name)
            plotModel = BoxPlotModel(analysis=wrapper, user=wrapper.user,
                                     name="BoxPlot {}".format(peak_detection_method_name),
                                     figure=box_png,
                                     based_on_peak_detection_method_name=peak_detection_method_name,
                                     based_on_performance_measure_name=performance_measure_method_name,
                                     based_on_peak_id=peak_id,
                                     )
            plotModel.save()  #
        self.update_progress(2, total_progress)

        # render decision trees
        tree_plot_tuples = TreePlot.DecisionTrees(mcc_ims_analysis.analysis_result, plot_parameters=plot_params)
        print(f"Saving {tree_plot_tuples}")
        for plot_tuple in tree_plot_tuples:
            for (eval_method_name, pdm_name, class_comparison_str), tree_plot_buffer, fig_name in plot_tuple:
                # print(eval_method_name, pdm_name, class_comparison_str, fig_name)
                dt_png = ImageFile(tree_plot_buffer, name=fig_name)
                plotModel = DecisionTreePlotModel(
                    analysis=wrapper, user=wrapper.user,
                    name=f"DecisionTree {pdm_name} {eval_method_name}", figure=dt_png,
                    based_on_peak_detection_method_name=pdm_name,
                    based_on_performance_measure_name=eval_method_name,
                )
                plotModel.save()

        # prebuild empty stats_dict for FDR - stats are only computed in random forest and cross validation
        # mcc_ims_analysis.analysis_result.pvalues_df
        self.update_progress(3, total_progress)

        for eval_method in performance_measures:
            if isinstance(eval_method, str):
                eval_method_name = eval_method
            elif isinstance(eval_method, PerformanceMeasure):
                eval_method_name = eval_method.name
            else:
                eval_method_name = str(eval_method)

            if eval_method_name == "DECISION_TREE_TRAINING":
                # don't do anything - just used to pass along stuff, not actually measure performance
                pass
            else:
                try:

                    stats_dict = {}
                    if eval_method_name != PerformanceMeasure.FDR_CORRECTED_P_VALUE.name:
                        stats_dict = \
                        mcc_ims_analysis.analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection[
                            eval_method_name]

                        #  needed to sanitize statsdict in multiclass setting
                        # for pdmn, stats in stats_dict.items():
                        #     # this is our problem
                        #     stats['auc_measures'].keys()
                        #     stats['auc_measures']['tpr_of_splits']
                        #
                        #     stats_dict.keys()


                    buffer = StringIO()
                    # get best_features from analysis result, grouped by eval_method
                    best_features_df = mcc_ims_analysis.analysis_result.best_features_df.loc[
                        mcc_ims_analysis.analysis_result.best_features_df['performance_measure_name'] == eval_method_name]
                    best_features_df.to_csv(buffer, index=True, header=True, index_label="index")
                    buffer.seek(0)
                    stats_model = StatisticsModel(
                        evaluation_method_name=eval_method_name,
                        analysis=wrapper,
                        statistics_dict=stats_dict,
                        best_features_df=ContentFile(buffer.getvalue(), name="best_features_df_{}.csv".format(eval_method_name))
                    )
                    buffer.close()
                    stats_model.save()
                    # print(stats_model.best_features_df)
                except TypeError:
                    # used to fail with Type error for json and ndarray - keep as workaround but pass
                    pass

        self.update_progress(4, total_progress)

        # make Overlay plots from best features
        # best_overlay_plot_png_tuples = ClusterPlot.OverlayBestFeaturesAlignment(mcc_ims_analysis,
        best_overlay_plot_png_tuples = ClusterPlot.OverlayBestFeaturesClasswiseAlignment(mcc_ims_analysis,
                                                                                 plot_parameters=plot_params)

        # no longer doing it for every measurement, but classwise instead - so measurement_name will be classwise_ sth
        for (peak_detection_method_name, measurement_name,
             performance_measure_name), best_overlay_png_buffer, fig_name in best_overlay_plot_png_tuples:
            best_overlay_png = ImageFile(best_overlay_png_buffer, name=fig_name)
            plotModel = BestFeaturesOverlayPlot(
                analysis=wrapper, user=wrapper.user,
                name="BestOverlay {} {}".format(peak_detection_method_name, performance_measure_name),
                figure=best_overlay_png,
                based_on_measurement=measurement_name,
                based_on_peak_detection_method_name=peak_detection_method_name,
                based_on_performance_measure_name=performance_measure_name,
            )
            plotModel.save()

        # we can also create decision trees as predictors and evaluate their performance - using RFC only atm
        # export predictor
        predictor = mcc_ims_analysis.analysis_result.export_prediction_models(path_to_save="", use_buffer=True)
        # print(predictor)
        class_labels = np.unique([k for k in class_counts.keys()]).tolist()

        # create and save prediction model to db
        # print(mcc_ims_analysis.analysis_result.feature_names_by_pdm)

        web_prediction_model = WebPredictionModel(
            name="PredictionModel {}".format(mcc_ims_analysis.dataset_name),
            scipy_predictor_pickle=ContentFile(predictor,
                                               name="predictor_pickle_{}".format(mcc_ims_analysis.dataset_name)),
            feature_names_by_pdm=mcc_ims_analysis.analysis_result.feature_names_by_pdm,
            mcc_ims_analysis=wrapper,
            class_labels=class_labels)
        web_prediction_model.save()
        # return web_prediction_model.pk
        self.update_progress(total_progress, total_progress)

        # need to create result id for show in result window and querying
        # prediction_model_pk = wrapper.evaluate_performance()
        return web_prediction_model.pk


class ParallelPreprocessingTask(JobtasticTask):
    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1

    def calculate_result(self, analysis_id, **kwargs):

        t0 = time.time()

        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        self.update_progress(1, 100)

        raw_file_ids = wrapper._unpack_raw_files_from_zip()

        number_of_measurements = len(raw_file_ids)
        # 1/3 peak detection, 1/3 heatmaps, 1/3 overlay plots
        total_progress = 3 * number_of_measurements

        self.update_progress(1, total_progress)

        # progress_recorder.set_progress(2, total_progress)

        preprocessing_options = wrapper.preprocessing_options

        # use chain to preprocess - first we need all peak detection results, then we can start plotting
        # raw_id_processed_id_pdr_id_tuples = []

        # res = celery_chain(
            # starts the parallel processing task and returns the raw_ids, processed_ids and pdr_ids
        prep_res = group(preprocess_measurement.s(raw_file_id, preprocessing_options) for raw_file_id in raw_file_ids)()
        # dangerous - in case we have too many ParallelPreprocessing tasks started at once - we will wait forever - so we fail after 30 minutes of preprocessing
        # and important to run celery with -Ofair option - workers kept stalling
        while not prep_res.successful():

            self.update_progress(prep_res.completed_count(), total_progress)
            time.sleep(1)

            if (time.time() - t0) > 120*60.:
                raise ValueError("Running for over 2 hours in pre-processing. Something is off.")

        self.update_progress(number_of_measurements, total_progress)
        raw_id_processed_id_pdr_id_tuples = prep_res.join()
        # print(raw_id_processed_id_pdr_id_tuples)

            # update_fileset.s(analysis_id=analysis_id),
            # heatmap_plots.s(),
            # cluster_overlay_plots.s(),
        # )()

        # for i, raw_file_id in enumerate(raw_file_ids):
        #     raw_id_processed_id_pdr_id_tuples.append(preprocess_measurement(raw_file_id, preprocessing_options))
        #     self.update_progress(i + 1, total_progress)
        update_fileset(tuples=raw_id_processed_id_pdr_id_tuples, analysis_id=analysis_id)

        time.sleep(1)
        self.update_progress(number_of_measurements + 2, total_progress)

        unzipped = list(zip(*raw_id_processed_id_pdr_id_tuples))
        # print(unzipped)
        # raw_ids = unzipped[0]
        processed_ids = unzipped[1]

        user_id = wrapper.user.pk
        # processed_file_ids = wrapper.ims_set.file_set.processed_files
        processed_file_ids = processed_ids
        print("Starting plotting for analysis {}\n Measurements: {}".format(analysis_id, processed_file_ids))
        # cluster_ids, overlay_ids = cluster_overlay_plots.s(analysis_id)()
        print("processed_file_ids", processed_file_ids)

        plot_group = celery_chain(
            group(heatmap_plot.s(pf_id, analysis_id, user_id) for pf_id in processed_file_ids),
            group(classwise_heatmap_plots.s(analysis_id=analysis_id, user_id=user_id), cluster_overlay_plots.s(analysis_id=analysis_id)))()
        # dangerous - in case we have too many ParallelPreprocessing tasks started at once - we will wait forever - so we fail after 30 minutes of preprocessing
        # and important to run celery with -Ofair option - workers kept stalling
        while not plot_group.successful():

            self.update_progress(number_of_measurements + 2 + prep_res.completed_count(), total_progress)
            time.sleep(1)

            if (time.time() - t0) > 120*60.:
                raise ValueError("Running for over 2 hours in pre-processing. Something is off.")


        # for i, pf_id in enumerate(processed_file_ids):
        #     heatmap_plot(pf_id, analysis_id, user_id)
        # self.update_progress(2 * number_of_measurements + 2 , total_progress)

        # make a classwise heatmap
        # classwise_heatmap_plots(analysis_id=analysis_id, user_id=user_id)

        # heatmap_plots(analysis_id=analysis_id)
        # self.update_progress(2 * number_of_measurements, total_progress)
        # update_progress.s(self, 4, total_progress),
        # cluster_overlay_plots(analysis_id=analysis_id)

        self.update_progress(total_progress, total_progress)
        return analysis_id


class CustomPeakDetectionAnalysisTask(JobtasticTask):
    """
    Do cross validation and feature extraction of Peak Detection results
    """
    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1

    def calculate_result(self, analysis_id, **kwargs):
        from collections import Counter
        from .models import WebPredictionModel, BoxPlotModel, RocPlotModel
        from breathpy.model.BreathCore import FeatureReductionMethod, PerformanceMeasure
        from breathpy.view.BreathVisualizations import RocCurvePlot, BoxPlot, TreePlot, \
            ClusterPlot
        import numpy as np


        self.update_progress(1, 100)


        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        number_of_measurements = len(wrapper.custom_set.file_set.peak_detection_results)
        total_progress = 4*number_of_measurements
        self.update_progress(1, total_progress)

        custom_fileset_id = wrapper.custom_set.file_set.pk
        class_label_mapping = wrapper.class_label_mapping
        analysis, plot_params, file_params = wrapper.reinitialize_mcc_ims_analysis(custom_fileset_id=custom_fileset_id)

        # problem here is that we dont have processed_measurements -
        # start analysis
        analysis.align_peaks(file_prefix="")

        # need to include this by default - alignment produces number of grid squares many peak_ids
        analysis.reduce_features(analysis.AVAILABLE_FEATURE_REDUCTION_METHODS)

        # get number of measurements from smallest class - need at least 2 measurements for cross validation
        class_counts = Counter(class_label_mapping.values())
        min_occurence = min(class_counts.values())
        can_cross_validate = min_occurence >= 10

        # get Performance measures directly from analysis object
        performance_measures = []
        for eo in wrapper.evaluation_options.keys():
            try:
                # we also have feature reduction options in evaluation_options
                performance_measures.append(PerformanceMeasure(eo))
            except ValueError:
                pass

        self.update_progress(number_of_measurements, total_progress)
        # print(mcc_ims_analysis)
        print(f"performance_measures = {performance_measures}")
        # print(self.evaluation_options)
        # pycharm confuses the two evaluate_performance functions
        analysis.evaluate_performance()

        self.update_progress(2 * number_of_measurements, total_progress)


        # only export trainings_matrix of best_model

        best_model_pdmname, feature_names, decision_tree_buffer = analysis.get_best_model()

        # update the analysis to only include best option
        preprocessing_options = wrapper.preprocessing_options
        # need to remove all other peak_detection_methods on initialization
        all_available_options = list(analysis.AVAILABLE_PEAK_DETECTION_METHODS)
        all_available_options.extend(analysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS)
        all_available_option_names = [ao.name for ao in all_available_options]

        keys_to_pop = set(all_available_option_names).difference(set([best_model_pdmname]))
        for ktp in keys_to_pop:
            preprocessing_options.pop(ktp, None)
        wrapper.preprocessing_options = preprocessing_options
        wrapper.automatic_selected_method_name = best_model_pdmname
        wrapper.save()
        print(f"Updated preprocessing options to {preprocessing_options}")

        # create Plots for best model

        # export feature matrices
        for pdm_name, trainings_matrix in analysis.analysis_result.trainings_matrix.items():

            fm_fn = "{}_feature_matrix".format(pdm_name)
            buff = StringIO()
            trainings_matrix.to_csv(buff, index=True, header=True, index_label="index")
            buff.seek(0)
            fm = FeatureMatrix(analysis=wrapper,
                               name=fm_fn,
                               peak_detection_method_name=pdm_name,
                               file=ContentFile(buff.getvalue(), name=fm_fn + ".csv"))
            fm.save()

        if can_cross_validate:
            # roc plots can only be done if we applied cross validation
            if len(set(analysis.analysis_result.class_labels)) == 2:
                roc_plot_tuples = RocCurvePlot.ROCCurve(
                    analysis.analysis_result, plot_parameters=plot_params,
                    limit_to_peak_detection_method_name=best_model_pdmname)
            else:
                roc_plot_tuples = RocCurvePlot.MultiClassROCCurve(analysis.analysis_result,
                                                                  plot_parameters=plot_params,
                                                                  limit_to_peak_detection_method_name=best_model_pdmname)

            for evaluation_method_name, peak_detection_method_name, roc_plot_buffer, fig_name in roc_plot_tuples:
                roc_png = ImageFile(roc_plot_buffer, name=fig_name)
                plotModel = RocPlotModel(analysis=wrapper, user=wrapper.user,
                                         name="Roc {}".format(peak_detection_method_name),
                                         based_on_peak_detection_method_name=peak_detection_method_name,
                                         based_on_performance_measure_name=evaluation_method_name,
                                         figure=roc_png)
                plotModel.save()

        # boxplots are only created if we have corrected pvalues
        # boxplots with new layout

        box_plot_tuples = BoxPlot.BoxPlotBestFeature(analysis.analysis_result, plot_parameters=plot_params,
                                                     limit_to_peak_detection_method_name=best_model_pdmname)
        for (performance_measure_method_name, peak_detection_method_name, model_of_class,
             peak_id), box_plot_buffer, fig_name in box_plot_tuples:
            box_png = ImageFile(box_plot_buffer, name=fig_name)
            plotModel = BoxPlotModel(analysis=wrapper, user=wrapper.user,
                                     name="BoxPlot {}".format(peak_detection_method_name),
                                     figure=box_png,
                                     based_on_peak_detection_method_name=peak_detection_method_name,
                                     based_on_performance_measure_name=performance_measure_method_name,
                                     based_on_peak_id=peak_id,
                                     )
            plotModel.save()  #

        self.update_progress(3 * number_of_measurements, total_progress)
        # self.update_progress(2, total_progress)

        # render decision trees
        tree_plot_tuples = TreePlot.DecisionTrees(analysis.analysis_result, plot_parameters=plot_params,
                                                  limit_to_peak_detection_method_name=best_model_pdmname)
        for plot_tuple in tree_plot_tuples:
            for (eval_method_name, pdm_name, class_comparison_str), tree_plot_buffer, fig_name in plot_tuple:
                # print(eval_method_name, pdm_name, class_comparison_str, fig_name)
                dt_png = ImageFile(tree_plot_buffer, name=fig_name)
                plotModel = DecisionTreePlotModel(
                    analysis=wrapper, user=wrapper.user,
                    name="DecisionTree {} {}".format(pdm_name, eval_method_name), figure=dt_png,
                    based_on_peak_detection_method_name=pdm_name,
                    based_on_performance_measure_name=eval_method_name,
                )
                plotModel.save()  #

        # prebuild empty stats_dict for FDR - stats are only computed in random forest and cross validation
        # mcc_ims_analysis.analysis_result.pvalues_df
        # self.update_progress(3, total_progress)

        for eval_method in performance_measures:
            if isinstance(eval_method, str):
                eval_method_name = eval_method
            elif isinstance(eval_method, PerformanceMeasure):
                eval_method_name = eval_method.name
            else:
                eval_method_name = str(eval_method)

            if eval_method_name == "DECISION_TREE_TRAINING":
                # don't do anything - just used to pass along stuff, not actually measure performance
                pass
            else:

                stats_dict = {}
                if eval_method_name != PerformanceMeasure.FDR_CORRECTED_P_VALUE.name:
                    stats_dict = \
                        analysis.analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection[
                            eval_method_name]

                buffer = StringIO()
                # get best_features from analysis result, grouped by eval_method
                best_features_df = analysis.analysis_result.best_features_df.loc[
                    analysis.analysis_result.best_features_df['performance_measure_name'] == eval_method_name]
                best_features_df.to_csv(buffer, index=True, header=True, index_label="index")
                buffer.seek(0)
                stats_model = StatisticsModel(
                    evaluation_method_name=eval_method_name,
                    analysis=wrapper,
                    statistics_dict=stats_dict,
                    best_features_df=ContentFile(buffer.getvalue(), name="best_features_df_{}.csv".format(eval_method_name))
                )
                buffer.close()
                stats_model.save()
                # print(stats_model.best_features_df)

        self.update_progress(total_progress - 1, total_progress)
        # self.update_progress(4, total_progress)

        # export predictor
        # predictors = mcc_ims_analysis.analysis_result.export_prediction_models(path_to_save="", use_buffer=True)
        # print(predictor)
        class_labels = np.unique([k for k in class_counts.keys()]).tolist()

        # create and save prediction model to db
        # print(mcc_ims_analysis.analysis_result.feature_names_by_pdm)
        # from .models import SingleWebPredictionModel
        web_prediction_model = WebPredictionModel(
            name="PredictionModel {} {}".format(analysis.dataset_name, best_model_pdmname),
            scipy_predictor_pickle=ContentFile(decision_tree_buffer,
                                               name="predictor_pickle_{}_{}".format(analysis.dataset_name,
                                                                                    best_model_pdmname)),
            feature_names_by_pdm={best_model_pdmname: feature_names},
            mcc_ims_analysis=wrapper,
            class_labels=class_labels,
        )
        web_prediction_model.save()
        # return web_prediction_model.pk
        self.update_progress(total_progress, total_progress)

        return analysis_id


class CustomPeakDetectionFMAnalysisTask(JobtasticTask):
    """
    Do cross validation and feature extraction of Peak Detection results
    """
    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1

    def calculate_result(self, analysis_id, **kwargs):
        from io import StringIO
        from collections import Counter
        from django.core.files.base import ContentFile
        from django.core.files.images import ImageFile
        from .models import WebPredictionModel, BoxPlotModel, RocPlotModel, FeatureMatrix
        from breathpy.model.ProcessingMethods import FeatureReductionMethod, PerformanceMeasure, ExternalPeakDetectionMethod
        from breathpy.view.BreathVisualizations import RocCurvePlot, BoxPlot, TreePlot
        import numpy as np


        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        is_custom_feature_matrix_analysis = wrapper.is_custom_feature_matrix_analysis
        if not is_custom_feature_matrix_analysis:
            raise ValueError("Calling wrong task.")

        feature_matrix_model = FeatureMatrix.objects.get(analysis=analysis_id, is_training_matrix=True)
        trainings_matrix = feature_matrix_model.get_feature_matrix()

        total_progress = 5
        self.update_progress(1, total_progress)

        # now evaluate with trainings matrix - skiping peak alignment, feature reduction, all that stuff
        # only do plots + roc / p_value analysis

        # only need AnalysisResult and class labels

        analysis, plot_params, file_params, coerced_pdm = wrapper.prepare_custom_fm_approach(feature_matrix_model)

        if FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES in analysis.AVAILABLE_FEATURE_REDUCTION_METHODS:

            # noise threshold works for peax - as z-normalized values are used to create mask
            reduced_fm = analysis.analysis_result.remove_redundant_features_fm(
                feature_matrix={coerced_pdm.name: trainings_matrix},
                class_label_dict=analysis.get_class_label_dict(),
                # use parameters from initialization
                **analysis.performance_measure_parameter_dict.get(FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES,
                                                                  {}))
        else:
            reduced_fm = {coerced_pdm.name: trainings_matrix}

        analysis.analysis_result.trainings_matrix = reduced_fm

        # get Performance measures directly from analysis object
        performance_measures = []
        for eo in wrapper.evaluation_options.keys():
            try:
                # we also have feature reduction options in evaluation_options
                performance_measures.append(PerformanceMeasure(eo))
            except ValueError:
                pass

        # pycharm confuses the two evaluate_performance functions - will use the one in BreathCore.MccImsAnalysis
        analysis.evaluate_performance(exising_analysis_result=analysis.analysis_result)


        # only export trainings_matrix of best_model - it's the only model trained
        best_model_pdmname, feature_names, decision_tree_buffer = analysis.get_best_model()

        # update the analysis to only include best option
        preprocessing_options = wrapper.preprocessing_options
        # need to remove all other peak_detection_methods on initialization
        all_available_options = list(analysis.AVAILABLE_PEAK_DETECTION_METHODS)
        all_available_options.extend(analysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS)
        all_available_option_names = [ao.name for ao in all_available_options]

        keys_to_pop = set(all_available_option_names).difference(set([best_model_pdmname]))
        for ktp in keys_to_pop:
            preprocessing_options.pop(ktp, None)
        wrapper.preprocessing_options = preprocessing_options
        wrapper.automatic_selected_method_name = best_model_pdmname
        wrapper.save()
        print(f"Updated preprocessing options to {preprocessing_options}")

        # create Plots for best model
        # print(f"We have {len(analysis.analysis_result.trainings_matrix)} feature matrices")
        # export feature matrices - not required! we start from a feature matrix! no need to duplicate it
        # for pdm_name, trainings_matrix in analysis.analysis_result.trainings_matrix.items():
        #     fm_fn = "{}_feature_matrix".format(pdm_name)
        #     buff = StringIO()
        #     from celery.contrib import rdb; rdb.set_trace()
        #     trainings_matrix.to_csv(buff, index=True, header=True, index_label="index")
        #     buff.seek(0)
        #     fm = FeatureMatrix(analysis=wrapper,
        #                        name=fm_fn,
        #                        peak_detection_method_name=pdm_name,
        #                        file=ContentFile(buff.getvalue(), name=fm_fn + ".csv"))
        #     fm.save()

        # get number of measurements from smallest class - need at least 2 measurements for cross validation
        class_counts = Counter([class_label for m_name, class_label in analysis.get_class_label_dict().items()])
        min_occurence = min(class_counts.values())
        can_cross_validate = min_occurence >= 10

        # cross validation is a given
        # roc plots can only be done if we applied cross validation
        if can_cross_validate:
            if len(set(analysis.analysis_result.class_labels)) == 2:
                roc_plot_tuples = RocCurvePlot.ROCCurve(
                    analysis.analysis_result, plot_parameters=plot_params,
                    limit_to_peak_detection_method_name=best_model_pdmname)
            else:
                roc_plot_tuples = RocCurvePlot.MultiClassROCCurve(analysis.analysis_result,
                                                                  plot_parameters=plot_params,
                                                                  limit_to_peak_detection_method_name=best_model_pdmname)

            for evaluation_method_name, peak_detection_method_name, roc_plot_buffer, fig_name in roc_plot_tuples:
                roc_png = ImageFile(roc_plot_buffer, name=fig_name)
                plotModel = RocPlotModel(analysis=wrapper, user=wrapper.user,
                                         name="Roc {}".format(peak_detection_method_name),
                                         based_on_peak_detection_method_name=peak_detection_method_name,
                                         based_on_performance_measure_name=evaluation_method_name,
                                         figure=roc_png)
                plotModel.save()

        # boxplots are only created if we have corrected pvalues
        # advanced boxplots here
        box_plot_tuples = BoxPlot.BoxPlotBestFeature(analysis.analysis_result, plot_parameters=plot_params,
                                                     limit_to_peak_detection_method_name=best_model_pdmname)
        for (performance_measure_method_name, peak_detection_method_name, model_of_class,
             peak_id), box_plot_buffer, fig_name in box_plot_tuples:
            box_png = ImageFile(box_plot_buffer, name=fig_name)
            plotModel = BoxPlotModel(analysis=wrapper, user=wrapper.user,
                                     name="BoxPlot {}".format(peak_detection_method_name),
                                     figure=box_png,
                                     based_on_peak_detection_method_name=peak_detection_method_name,
                                     based_on_performance_measure_name=performance_measure_method_name,
                                     based_on_peak_id=peak_id,
                                     )
            plotModel.save()  #

        self.update_progress(2, total_progress)

        # render decision trees
        tree_plot_tuples = TreePlot.DecisionTrees(analysis.analysis_result, plot_parameters=plot_params,
                                                  limit_to_peak_detection_method_name=best_model_pdmname)
        for plot_tuple in tree_plot_tuples:
            for (eval_method_name, pdm_name, class_comparison_str), tree_plot_buffer, fig_name in plot_tuple:
                # print(eval_method_name, pdm_name, class_comparison_str, fig_name)
                dt_png = ImageFile(tree_plot_buffer, name=fig_name)
                plotModel = DecisionTreePlotModel(
                    analysis=wrapper, user=wrapper.user,
                    name="DecisionTree {} {}".format(pdm_name, eval_method_name), figure=dt_png,
                    based_on_peak_detection_method_name=pdm_name,
                    based_on_performance_measure_name=eval_method_name,
                )
                plotModel.save()  #

        # prebuild empty stats_dict for FDR - stats are only computed in random forest and cross validation
        # mcc_ims_analysis.analysis_result.pvalues_df
        self.update_progress(3, total_progress)

        for eval_method in performance_measures:
            if isinstance(eval_method, str):
                eval_method_name = eval_method
            elif isinstance(eval_method, PerformanceMeasure):
                eval_method_name = eval_method.name
            else:
                eval_method_name = str(eval_method)

            if eval_method_name == "DECISION_TREE_TRAINING":
                # don't do anything - just used to pass along stuff, not actually measuring performance for DT
                pass
            else:

                stats_dict = {}
                if eval_method_name != PerformanceMeasure.FDR_CORRECTED_P_VALUE.name:
                    stats_dict = \
                        analysis.analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection[
                            eval_method_name]

                buffer = StringIO()
                # get best_features from analysis result, grouped by eval_method
                best_features_df = analysis.analysis_result.best_features_df.loc[
                    analysis.analysis_result.best_features_df['performance_measure_name'] == eval_method_name]
                best_features_df.to_csv(buffer, index=True, header=True, index_label="index")
                buffer.seek(0)
                stats_model = StatisticsModel(
                    evaluation_method_name=eval_method_name,
                    analysis=wrapper,
                    statistics_dict=stats_dict,
                    best_features_df=ContentFile(buffer.getvalue(), name="best_features_df_{}.csv".format(eval_method_name))
                )
                buffer.close()
                stats_model.save()
                # print(stats_model.best_features_df)

        self.update_progress(4, total_progress)

        # export predictor
        # predictors = mcc_ims_analysis.analysis_result.export_prediction_models(path_to_save="", use_buffer=True)
        # print(predictor)
        class_labels = np.unique(list(analysis.class_label_dict.values())).tolist()

        # create and save prediction model to db
        # print(mcc_ims_analysis.analysis_result.feature_names_by_pdm)
        # from .models import SingleWebPredictionModel
        web_prediction_model = WebPredictionModel(
            name=f"PredictionModel {analysis.dataset_name} {best_model_pdmname}",
            scipy_predictor_pickle=ContentFile(decision_tree_buffer,
                                               name=f"predictor_pickle_{analysis.dataset_name}_{best_model_pdmname}"),
            feature_names_by_pdm={best_model_pdmname: feature_names},
            mcc_ims_analysis=wrapper,
            class_labels=class_labels)
        web_prediction_model.save()
        self.update_progress(total_progress, total_progress)

        return analysis_id


class CustomPredictClassTask(JobtasticTask):
    """
    This task uses a prediction model and predicts the classes of pdr
    The prediction model requires a Float / BitPeakAlignmentResult with the same peak_ids -
      so we need to reconstruct / use the same columns from model training
    """

    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1


    def calculate_result(self, analysis_id, customPDFS_id=0, feature_matrix_id=0, **kwargs):
        from .models import WebPredictionModel, PredictionResult, FeatureMatrix
        from breathpy.model.BreathCore import PredictionModel
        import joblib
        total_progress = 3

        # get evaluation params from analysis
        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        web_prediction_model = WebPredictionModel.objects.get(mcc_ims_analysis=analysis_id)

        # use preprocess measurement method
        # give peak detection results to prediction
        self.update_progress(1, total_progress)

        using_custom_feature_matrix = False
        if customPDFS_id:
            customPDFS = CustomPeakDetectionFileSet.objects.get(pk=customPDFS_id)
            mcc_ims_analysis, plot_parameters, file_parameters = wrapper.reinitialize_mcc_ims_analysis(
                custom_fileset_id=customPDFS_id)

            # align_peaks will use the standard grid
            mcc_ims_analysis.align_peaks(file_prefix="")
            # no feature reduction applied - because we reduce to trainings columns
            # make sure features match
            df_matching_training = PredictionModel.reconstruct_remove_features(
                mcc_ims_analysis.peak_alignment_result.dict_of_df, web_prediction_model.feature_names_by_pdm)

            self.update_progress(2, total_progress)

            scipy_predictor_by_pdm = joblib.load(web_prediction_model.scipy_predictor_pickle.path)

            # get original labels
            original_labels = customPDFS.get_class_label_processed_id_dict()

        else:
            feature_matrix = FeatureMatrix.objects.get(pk=feature_matrix_id)
            using_custom_feature_matrix = True
            mcc_ims_analysis, plot_parameters, file_parameters, coerced_pdm = wrapper.prepare_custom_fm_approach(
                feature_matrix_model=feature_matrix)
            # no alignment

            # make sure features match
            df_matching_training = PredictionModel.reconstruct_remove_features(
                mcc_ims_analysis.analysis_result.trainings_matrix, web_prediction_model.feature_names_by_pdm)

            self.update_progress(2, total_progress)

            scipy_predictor_by_pdm = joblib.load(web_prediction_model.scipy_predictor_pickle.path)

            # get original labels
            original_labels = feature_matrix.get_class_label_dict()

        # print(f"original_labels: {original_labels}")

        # get labels to assign class and return prediction
        # get evaluation matrix for each chosen peak_detection method
        prediction_by_pdm = dict()
        for pdm in mcc_ims_analysis.peak_detection_combined:
            X_reduced = df_matching_training[pdm.name]
            # Import exisitng model and predict classes of unknown data
            # probas = self.scipy_predictor.predict_proba(X)
            prediction_matrix = X_reduced.fillna(0)

            buff = StringIO()
            prediction_matrix.to_csv(buff, index=True, header=True, index_label="index")
            buff.seek(0)
            pm_fn = f"prediction_matrix_{pdm.name}.csv"

            # create FeatureMatrix and save in backend for potential export
            prediction_matrix_model = FeatureMatrix(
                    analysis=wrapper, name=f"Prediction Matrix {pdm.name}",
                    peak_detection_method_name=pdm.name,
                    file=ContentFile(buff.getvalue(), name=pm_fn),
                    is_training_matrix=False, is_prediction_matrix=True)
            prediction_matrix_model.save()

            prediction_by_pdm[pdm] = scipy_predictor_by_pdm[pdm].predict(prediction_matrix).tolist()

        # print(f"prediction_by_pdm[pdm] {prediction_by_pdm[pdm]}")

        labeled_prediction_result = {}

        for pdm, prediction_result in prediction_by_pdm.items():
            # get measurement_names from feature_matrix index
            measurement_names = df_matching_training[pdm.name].index
            labeled_prediction_result[pdm] = {measurement_name: web_prediction_model.class_labels[class_index] for
                                              measurement_name, class_index in
                                              zip(measurement_names, prediction_result)}
            # print(f"labeled_prediction_result[pdm] {labeled_prediction_result[pdm]}")
            # print(f"prediction_result {prediction_result}")

        for pdm, prediction_result in labeled_prediction_result.items():
            prediction_result_model = PredictionResult(
                web_prediction_model=web_prediction_model,
                class_assignment=prediction_result,
                peak_detection_method_name=pdm.name,
                original_class_labels=original_labels,
            )
            prediction_result_model.save()

        self.update_progress(total_progress, total_progress)
        return web_prediction_model.pk


# class PreprocessingTask(JobtasticTask):
#     significant_kwargs = [('analysis_id', str)]
#     herd_avoidance_timeout = -1
#     cache_duration = -1
#
#     def calculate_result(self, analysis_id, **kwargs):
#
#         wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)
#
#         raw_file_ids = wrapper._unpack_raw_files_from_zip()
#
#         number_of_measurements = len(raw_file_ids)
#         # 1/3 peak detection, 1/3 heatmaps, 1/3 overlay plots
#         total_progress = 3*number_of_measurements
#
#         self.update_progress(1, total_progress)
#
#         # progress_recorder.set_progress(2, total_progress)
#
#         preprocessing_options = wrapper.preprocessing_options
#
#         # use chain to preprocess - first we need all peak detection results, then we can start plotting
#         raw_id_processed_id_pdr_id_tuples = []
#         for i, raw_file_id in enumerate(raw_file_ids):
#             raw_id_processed_id_pdr_id_tuples.append(preprocess_measurement(raw_file_id, preprocessing_options))
#             self.update_progress(i+1, total_progress)
#         update_fileset(tuples=raw_id_processed_id_pdr_id_tuples, analysis_id=analysis_id)
#
#         time.sleep(1)
#         self.update_progress(number_of_measurements+2, total_progress)
#
#         unzipped = list(zip(*raw_id_processed_id_pdr_id_tuples))
#         # print(unzipped)
#         # raw_ids = unzipped[0]
#         processed_ids = unzipped[1]
#
#         user_id = wrapper.user.pk
#         # processed_file_ids = wrapper.ims_set.file_set.processed_files
#         processed_file_ids = processed_ids
#         print("Starting plotting for analysis {}\n Measurements: {}".format(analysis_id, processed_file_ids))
#         # cluster_ids, overlay_ids = cluster_overlay_plots.s(analysis_id)()
#         print("processed_file_ids", processed_file_ids)
#         for i, pf_id in enumerate(processed_file_ids):
#             heatmap_plot(pf_id, analysis_id, user_id)
#             self.update_progress(number_of_measurements+2+i, total_progress)
#
#         # make a classwise heatmap
#         classwise_heatmap_plots(analysis_id=analysis_id, user_id=user_id)
#
#
#         # heatmap_plots(analysis_id=analysis_id)
#         self.update_progress(2*number_of_measurements, total_progress)
#             # update_progress.s(self, 4, total_progress),
#         cluster_overlay_plots(analysis_id=analysis_id)
#
#         self.update_progress(total_progress,total_progress)
#         return analysis_id

class ParallelAutomaticPreprocessingEvaluationTask(JobtasticTask):
    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1
    """
    Do automatic pipeline
    Check all peak detection methods and evaluation options
    Then only create Prediction Model for best predictor and keep features for it's evaluation - remove rest
    """

    def calculate_result(self, analysis_id, **kwargs):

        # PREPROCESSING and PEAK_DETECTION of measurements

        from collections import Counter
        from .models import WebPredictionModel, BoxPlotModel, RocPlotModel
        from breathpy.model.BreathCore import FeatureReductionMethod, PerformanceMeasure
        from breathpy.view.BreathVisualizations import RocCurvePlot, BoxPlot, TreePlot, \
            ClusterPlot
        import numpy as np

        t0 = time.time()

        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        raw_file_ids = wrapper._unpack_raw_files_from_zip()

        number_of_measurements = len(raw_file_ids)

        # 1/4 peak detection, 1/4 heatmaps, + initial overlay plots
        # 1/4 performance evaluation, 1/4 best Overlay Plots + boxplots + decision trees
        total_progress = 4*number_of_measurements

        self.update_progress(1, total_progress)

        # are set on creation
        preprocessing_options = wrapper.preprocessing_options


        # PREPROCESSING and PeakDetection
        # use chain to preprocess - first we need all peak detection results, then we can start plotting
        # raw_id_processed_id_pdr_id_tuples = []
        # for i, raw_file_id in enumerate(raw_file_ids):
        #     raw_id_processed_id_pdr_id_tuples.append(preprocess_measurement(raw_file_id, preprocessing_options))
        #     self.update_progress(i+1, total_progress)
        # update_fileset(tuples=raw_id_processed_id_pdr_id_tuples, analysis_id=analysis_id)

        # starts the parallel processing task and returns the raw_ids, processed_ids and pdr_ids
        prep_res = group(preprocess_measurement.s(raw_file_id, preprocessing_options) for raw_file_id in raw_file_ids)()
        # dangerous - in case we have too many ParallelPreprocessing tasks started at once - we will wait forever - so we fail after 30 minutes of preprocessing
        # and important to run celery with -Ofair option - workers kept stalling
        while not prep_res.successful():

            self.update_progress(prep_res.completed_count(), total_progress)
            time.sleep(1)

            if (time.time() - t0) > 120 * 60.:
                raise ValueError("Running for over 2 hours in pre-processing. Something is off.")

        self.update_progress(number_of_measurements, total_progress)

        # this can lead to a deadlock (eg if more than 5 users and just 10 tasks) - but we want the progress reporting
        raw_id_processed_id_pdr_id_tuples = prep_res.join()
        update_fileset(tuples=raw_id_processed_id_pdr_id_tuples, analysis_id=analysis_id)

        time.sleep(1)
        self.update_progress(number_of_measurements+2, total_progress)

        unzipped = list(zip(*raw_id_processed_id_pdr_id_tuples))
        # print(unzipped)
        # raw_ids = unzipped[0]
        processed_ids = unzipped[1]

        user_id = wrapper.user.pk
        # processed_file_ids = wrapper.ims_set.file_set.processed_files
        processed_file_ids = processed_ids

        print("Starting plotting for analysis {}\n Measurements: {}".format(analysis_id, processed_file_ids))
        # cluster_ids, overlay_ids = cluster_overlay_plots.s(analysis_id)()
        print("processed_file_ids", processed_file_ids)

        plot_group = celery_chain(
            group(heatmap_plot.s(pf_id, analysis_id, user_id) for pf_id in processed_file_ids),
            group(classwise_heatmap_plots.s(analysis_id=analysis_id, user_id=user_id),
                  cluster_overlay_plots.s(analysis_id=analysis_id)))()
        # dangerous - in case we have too many ParallelPreprocessing tasks started at once - we will wait forever - so we fail after 30 minutes of preprocessing
        # and important to run celery with -Ofair option - workers kept stalling
        while not plot_group.successful():

            self.update_progress(number_of_measurements + 2 + prep_res.completed_count(), total_progress)
            time.sleep(1)

            if (time.time() - t0) > 120 * 60.:
                raise ValueError("Running for over 2 hours in pre-processing. Something is off.")


        # for i, pf_id in enumerate(processed_file_ids):
        #     heatmap_plot(pf_id, analysis_id, user_id)
        #     self.update_progress(number_of_measurements+2+i, total_progress)

        # make a classwise heatmap
        # classwise_heatmap_plots(analysis_id=analysis_id, user_id=user_id)

        # heatmap_plots(analysis_id=analysis_id)
        # cluster_overlay_plots(analysis_id=analysis_id)

        self.update_progress(2*number_of_measurements, total_progress)

        #############################
        # Next - evaluate performance


        # only do plots of best model
        # # get evaluation params from analysis
        # wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)
        # # wrapper.ev wrapper.evaluation_options
        # total_progress = 5

        # Create BoxPlots, decision trees and export statistics csv
        # get MccImsAnalysis instance from wraper
        # get preprocessed files from DB
        print("Reinitializing mcc_ims_analysis")
        # re-getting is a requirement - as we get an outdated fileset if we dont do it this way
        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)
        mcc_ims_analysis, plot_params, file_params = wrapper.reinitialize_mcc_ims_analysis()

        # problem here is that we dont have processed_measurements -
        # start analysis
        # fileset = wrapper.ims_set.file_set
        # print(fileset)
        # print(fileset.peak_detection_results)
        mcc_ims_analysis.align_peaks(file_prefix="")

        mcc_ims_analysis.reduce_features(mcc_ims_analysis.AVAILABLE_FEATURE_REDUCTION_METHODS)

        # get number of measurements from smallest class - need at least 2 measurements for cross validation
        class_counts = Counter([m.class_label for m in mcc_ims_analysis.measurements])
        min_occurence = min(class_counts.values())
        can_cross_validate = min_occurence > 10

        # get Performance measures directly from analysis object
        performance_measures = []
        for eo in wrapper.evaluation_options.keys():
            try:
                # we also have feature reduction options in evaluation_options
                performance_measures.append(PerformanceMeasure(eo))
            except ValueError:
                pass

        self.update_progress(2*number_of_measurements +1, total_progress)
        # print(mcc_ims_analysis)
        print(f"performance_measures = {performance_measures}")
        # print(self.evaluation_options)
        # pycharm confuses the two evaluate_performance functions
        mcc_ims_analysis.evaluate_performance()

        self.update_progress(2.5 * number_of_measurements, total_progress)

        # only export trainings_matrix of best_model


        best_model_pdmname, feature_names, decision_tree_buffer = mcc_ims_analysis.get_best_model()


        # update the analysis to only include best option
        preprocessing_options = wrapper.preprocessing_options
        # need to remove all other peak_detection_methods on initialization
        all_available_options = list(mcc_ims_analysis.AVAILABLE_PEAK_DETECTION_METHODS)
        all_available_options.extend(mcc_ims_analysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS)
        all_available_option_names = [ao.name for ao in all_available_options]

        keys_to_pop = set(all_available_option_names).difference(set([best_model_pdmname]))
        for ktp in keys_to_pop:
            preprocessing_options.pop(ktp, None)
        wrapper.preprocessing_options = preprocessing_options
        wrapper.automatic_selected_method_name = best_model_pdmname
        wrapper.save()
        print(f"Updated preprocessing options to {preprocessing_options}")


        # create Plots for best model

        # export feature matrices
        for pdm_name, trainings_matrix in mcc_ims_analysis.analysis_result.trainings_matrix.items():
            fm_fn = "{}_feature_matrix".format(pdm_name)
            buff = StringIO()
            trainings_matrix.to_csv(buff, index=True, header=True, index_label="index")
            buff.seek(0)
            fm = FeatureMatrix(analysis=wrapper,
                               name=fm_fn,
                               peak_detection_method_name=pdm_name,
                               file=ContentFile(buff.getvalue(), name=fm_fn + ".csv"))
            fm.save()

        if can_cross_validate:
            # roc plots can only be done if we applied cross validation
            if len(set(mcc_ims_analysis.analysis_result.class_labels)) == 2:
                roc_plot_tuples = RocCurvePlot.ROCCurve(
                        mcc_ims_analysis.analysis_result, plot_parameters=plot_params, limit_to_peak_detection_method_name=best_model_pdmname)
            else:
                roc_plot_tuples = RocCurvePlot.MultiClassROCCurve(mcc_ims_analysis.analysis_result,
                                                                  plot_parameters=plot_params,
                                                                  limit_to_peak_detection_method_name=best_model_pdmname)

            for evaluation_method_name, peak_detection_method_name, roc_plot_buffer, fig_name in roc_plot_tuples:
                roc_png = ImageFile(roc_plot_buffer, name=fig_name)
                plotModel = RocPlotModel(analysis=wrapper, user=wrapper.user,
                                         name="Roc {}".format(peak_detection_method_name),
                                         based_on_peak_detection_method_name=peak_detection_method_name,
                                         based_on_performance_measure_name=evaluation_method_name,
                                         figure=roc_png)
                plotModel.save()

        # boxplots are only created if we have corrected pvalues
        box_plot_tuples = BoxPlot.BoxPlotBestFeature(mcc_ims_analysis.analysis_result, plot_parameters=plot_params,
                                                     limit_to_peak_detection_method_name=best_model_pdmname)
        for (performance_measure_method_name, peak_detection_method_name, model_of_class,
             peak_id), box_plot_buffer, fig_name in box_plot_tuples:
            box_png = ImageFile(box_plot_buffer, name=fig_name)
            plotModel = BoxPlotModel(analysis=wrapper, user=wrapper.user,
                                     name="BoxPlot {}".format(peak_detection_method_name),
                                     figure=box_png,
                                     based_on_peak_detection_method_name=peak_detection_method_name,
                                     based_on_performance_measure_name=performance_measure_method_name,
                                     based_on_peak_id=peak_id,
                                     )
            plotModel.save()  #

        self.update_progress(3*number_of_measurements, total_progress)
        # self.update_progress(2, total_progress)

        # render decision trees
        tree_plot_tuples = TreePlot.DecisionTrees(mcc_ims_analysis.analysis_result, plot_parameters=plot_params,
                                                  limit_to_peak_detection_method_name=best_model_pdmname)
        for plot_tuple in tree_plot_tuples:
            for (eval_method_name, pdm_name, class_comparison_str), tree_plot_buffer, fig_name in plot_tuple:
                # print(eval_method_name, pdm_name, class_comparison_str, fig_name)
                dt_png = ImageFile(tree_plot_buffer, name=fig_name)
                plotModel = DecisionTreePlotModel(
                    analysis=wrapper, user=wrapper.user,
                    name="DecisionTree {} {}".format(pdm_name, eval_method_name), figure=dt_png,
                    based_on_peak_detection_method_name=pdm_name,
                    based_on_performance_measure_name=eval_method_name,
                )
                plotModel.save()  #

        # prebuild empty stats_dict for FDR - stats are only computed in random forest and cross validation
        # mcc_ims_analysis.analysis_result.pvalues_df
        # self.update_progress(3, total_progress)

        for eval_method in performance_measures:
            if isinstance(eval_method, str):
                eval_method_name = eval_method
            elif isinstance(eval_method, PerformanceMeasure):
                eval_method_name = eval_method.name
            else:
                eval_method_name = str(eval_method)

            if eval_method_name == "DECISION_TREE_TRAINING":
                # don't do anything - just used to pass along stuff, not actually measure performance
                pass
            else:
                stats_dict = {}
                if eval_method_name != PerformanceMeasure.FDR_CORRECTED_P_VALUE.name:
                    stats_dict = \
                    mcc_ims_analysis.analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection[
                        eval_method_name]

                buffer = StringIO()
                # get best_features from analysis result, grouped by eval_method
                best_features_df = mcc_ims_analysis.analysis_result.best_features_df.loc[
                    mcc_ims_analysis.analysis_result.best_features_df['performance_measure_name'] == eval_method_name]
                best_features_df.to_csv(buffer, index=True, header=True, index_label="index")
                buffer.seek(0)
                stats_model = StatisticsModel(
                    evaluation_method_name=eval_method_name,
                    analysis=wrapper,
                    statistics_dict=stats_dict,
                    best_features_df=ContentFile(buffer.getvalue(), name="best_features_df_{}.csv".format(eval_method_name))
                )
                buffer.close()
                stats_model.save()
                # print(stats_model.best_features_df)

        self.update_progress(total_progress-2, total_progress)
        # self.update_progress(4, total_progress)

        # make Overlay plots from best featuresg
        # best_overlay_plot_png_tuples = ClusterPlot.OverlayBestFeaturesAlignment(mcc_ims_analysis,
        best_overlay_plot_png_tuples = ClusterPlot.OverlayBestFeaturesClasswiseAlignment(mcc_ims_analysis,
                                                                                 plot_parameters=plot_params)

        # no longer doing it for every measurement, but classwise instead - so measurement_name will be classwise_ sth
        for (peak_detection_method_name, measurement_name,
             performance_measure_name), best_overlay_png_buffer, fig_name in best_overlay_plot_png_tuples:
            best_overlay_png = ImageFile(best_overlay_png_buffer, name=fig_name)
            plotModel = BestFeaturesOverlayPlot(
                analysis=wrapper, user=wrapper.user,
                name="BestOverlay {} {}".format(peak_detection_method_name, performance_measure_name),
                figure=best_overlay_png,
                based_on_measurement=measurement_name,
                based_on_peak_detection_method_name=peak_detection_method_name,
                based_on_performance_measure_name=performance_measure_name,
            )
            plotModel.save()


        # export predictor
        # predictors = mcc_ims_analysis.analysis_result.export_prediction_models(path_to_save="", use_buffer=True)
        # print(predictor)
        class_labels = np.unique([k for k in class_counts.keys()]).tolist()

        # create and save prediction model to db
        # print(mcc_ims_analysis.analysis_result.feature_names_by_pdm)
        # from .models import SingleWebPredictionModel
        web_prediction_model = WebPredictionModel(
            name="PredictionModel {} {}".format(mcc_ims_analysis.dataset_name, best_model_pdmname),
            scipy_predictor_pickle=ContentFile(decision_tree_buffer,
                                               name="predictor_pickle_{}_{}".format(mcc_ims_analysis.dataset_name, best_model_pdmname)),
            feature_names_by_pdm={best_model_pdmname : feature_names},
            mcc_ims_analysis=wrapper,
            class_labels=class_labels)
        web_prediction_model.save()
        # return web_prediction_model.pk
        self.update_progress(total_progress,total_progress)

        return analysis_id


@app.task(bind=True)
def full_parallel_preprocessing(self, analysis_id):
    # could be used to do parallel preprocessing, but does not allow for progress tracking but will return emmidiately
    from breathpy.model.BreathCore import ExternalPeakDetectionMethod
    from shutil import rmtree
    # from functools import partial

    print("Starting preprocessing of measurements")

    wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

    raw_file_ids = wrapper._unpack_raw_files_from_zip()

    preprocessing_options = wrapper.preprocessing_options
    # use chain to preprocess - first we need all peak detection results, then we can start plotting
    res = celery_chain(
        # starts the parallel processing task and returns the raw_ids, processed_ids and pdr_ids
        group(preprocess_measurement.s(raw_file_id, preprocessing_options) for raw_file_id in raw_file_ids),
        update_fileset.s(analysis_id=analysis_id),
        heatmap_plots.s(),
        cluster_overlay_plots.s(),
    )()
    print("Parallel_preprocessing finished.")
    return res


@app.task
def update_fileset(tuples, analysis_id):
    """
    Update Analysiswrapper and associated fileset with preprocessing results
    :param tuples: tuples of ids, raw_file ids, processed file ids and wep_peak_detection_result_ids
    :param analysis_id:
    :return:
    """
    from breathpy.model.BreathCore import ExternalPeakDetectionMethod
    from shutil import rmtree

    wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)
    # user_id = wrapper.ims_set.user_id.pk
    archive_name = wrapper.ims_set.upload.name

    # [r_id, pr_id, [wpdr_id,], ... ]
    unzipped = list(zip(*tuples))
    print(unzipped)
    raw_ids = unzipped[0]
    processed_ids = unzipped[1]
    pdr_ids = list(chain.from_iterable(unzipped[2]))

    fileset = FileSet(name=archive_name[:50], raw_files=raw_ids, processed_files=processed_ids,
                      peak_detection_results=pdr_ids)
    fileset.save()

    print("Created Fileset {}".format(fileset.pk))
    # cleanup temporary files of peax directory
    if ExternalPeakDetectionMethod.PEAX.name in wrapper.preprocessing_options:
        tmp = wrapper.preprocessing_options[ExternalPeakDetectionMethod.PEAX.name]['tempdir']
        rmtree(tmp)

    # save label_id dict in model for use in consensus plot
    pr_files = ProcessedFile.objects.filter(pk__in=processed_ids)
    id_label_tuples = ((prf.label, prf.pk) for prf in pr_files)
    class_label_id_dict = defaultdict(list)
    for label, processed_id in id_label_tuples:
        class_label_id_dict[label].append(processed_id)

    fileset.class_label_processed_id_dict = class_label_id_dict
    fileset.save()

    # replace initial fileset with one containing all info
    wrapper.ims_set.file_set = fileset
    wrapper.ims_set.save()
    wrapper.save()
    print("Bound Fileset {} to ims_set {} to analysis {}".format(fileset.pk, wrapper.ims_set.pk, wrapper.pk))
    return analysis_id


@app.task
def heatmap_plots(analysis_id):
    """
    Make Heatmap plots of all preprocessed measurements associated with analysis_id.
    Starting a group - parallel tasks for each heatmap
    :param analysis_id:
    :return:
    """
    wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)
    user_id = wrapper.user.pk
    processed_file_ids = wrapper.ims_set.file_set.processed_files
    print("Starting parallel plotting for analysis {}\n Measurements: {}".format(analysis_id, processed_file_ids))
    # cluster_ids, overlay_ids = cluster_overlay_plots.s(analysis_id)()
    res = group(heatmap_plot.s(processed_file_id, analysis_id, user_id) for processed_file_id in processed_file_ids)()
    return analysis_id


@app.task
def preprocess_measurement(raw_file_id, preprocessing_options):
    """
    Apply all preprocessing options to raw files, create model instances and save to db.
    Normalization, Denoising and all PeakDetection is applied here
    :param raw_file_id:
    :param preprocessing_options:
    :return:
    """
    from breathpy.model.BreathCore import MccImsMeasurement, MccImsAnalysis, \
        ExternalPeakDetectionMethod
    # load measurement from db
    # preprocess measurement
    # save processed file in db
    # apply peak detection methods
    raw_file = RawFile.objects.get(pk=raw_file_id)
    measurement = MccImsMeasurement(raw_file.file.file)

    matched_steps = MccImsAnalysis.match_processing_options(preprocessing_options)

    # check if we have external peak detection -> external want raw measurements
    external_steps = matched_steps['external_steps']

    peax_pdr = None
    if ExternalPeakDetectionMethod.PEAX in external_steps:
        # call peax to handle export and reimport
        # peax is sometimes extremely slow - takes 5x as long as usual when running as async task
        peax_pdr = MccImsAnalysis.external_peak_detection_peax_helper(measurement, **preprocessing_options[
            ExternalPeakDetectionMethod.PEAX.name])
        # print(peax_pdr)

    measurement.normalize_measurement(matched_steps['normalization_steps'])
    measurement.denoise_measurement(matched_steps['denoising_steps'])

    processed_file = ProcessedFile(
        raw_file=raw_file,
        file=ContentFile(
            measurement.export_to_csv(use_buffer=True),
            name="{}_preprocessed.csv".format(measurement.filename[:-4])),
        label=raw_file.label,
    )
    processed_file.save()
    processed_file_id = processed_file.pk

    web_pdr_ids = []

    if ExternalPeakDetectionMethod.PEAX in external_steps:
        # assign and save processed_file id to peax result - even though we used raw
        result_filename, pdm_name = WebPeakDetectionResult.fields_from_peak_detection_result(peax_pdr)
        peax_wpdr = WebPeakDetectionResult(
            filename=result_filename,
            peak_detection_method_name=pdm_name,
            preprocessed_measurement=processed_file,
            csv_file=ContentFile(peax_pdr.export_as_csv(directory="", use_buffer=True), name=result_filename + ".csv")
        )
        peax_wpdr.save()
        web_pdr_ids.append(peax_wpdr.pk)

    # create WebPeakDetectionResult for each peak detection method, yield?
    for pdr in MccImsAnalysis.detect_peaks_helper(
            measurement=measurement,
            peak_detection_steps=matched_steps['peak_detection_steps'],
            preprocessing_parameter_dict=MccImsAnalysis.prepare_preprocessing_parameter_dict(preprocessing_options)
    ):

        try:
            result_filename, pdm_name = WebPeakDetectionResult.fields_from_peak_detection_result(pdr)
        except ValueError as ve:
            raise ve

        wpdr = WebPeakDetectionResult(
            filename=result_filename,
            peak_detection_method_name=pdm_name,
            preprocessed_measurement=processed_file,
            csv_file=ContentFile(pdr.export_as_csv(directory="", use_buffer=True), name=result_filename + ".csv")
        )
        wpdr.save()
        web_pdr_ids.append(wpdr.pk)
    return raw_file_id, processed_file_id, web_pdr_ids


@app.task
def cluster_overlay_plots(analysis_id):
    """
    Create all Cluster and OveralyPlots for peak detection results. Single threaded.
    Reinitialize MccImsAnalysis with peak detection results
    :param analysis_id:
    :return:
    """
    from breathpy.view.BreathVisualizations import ClusterPlot

    wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)
    print("Starting cluster overlay plots for analysis {}\n Measurements: {}".format(analysis_id,
                                                                                     wrapper.ims_set.file_set.processed_files))
    mcc_ims_analysis, plot_params, file_params = wrapper.reinitialize_mcc_ims_analysis()
    print("Finished reinit of analysis")
    mcc_ims_analysis.align_peaks(file_prefix="")
    print("Finished peak alignment ")
    cluster_plot_ids = []
    overlay_plot_ids = []
    # Create Cluster plots
    # distinguish different used methods by using tuples
    print("Starting cluster plots")
    cluster_png_tuples = ClusterPlot.ClusterBasic(mcc_ims_analysis, plot_parameters=plot_params)
    for clustering_method_name, peak_detection_method_name, cluster_png_buffer, fig_name in cluster_png_tuples:
        cluster_png = ImageFile(cluster_png_buffer, name=fig_name)
        plotModel = ClusterPlotModel(analysis=wrapper, user=wrapper.user,
                                     name="Cluster {}".format(peak_detection_method_name)[:50],
                                     based_on_peak_detection_method_name=peak_detection_method_name,
                                     based_on_peak_alignment_method_name=clustering_method_name,
                                     figure=cluster_png)
        plotModel.save()
        cluster_plot_ids.append(plotModel.pk)
    print("Finished cluster plots")
    print("Starting overlay plots")
    # Create overlay plots for all of the peak detection methods - but only one per prediction class
    # overlay_plot_png_tuples = ClusterPlot.OverlayAlignment(mcc_ims_analysis, plot_parameters=plot_params)
    overlay_plot_png_tuples = ClusterPlot.OverlayClasswiseAlignment(mcc_ims_analysis, plot_parameters=plot_params)
    for peak_detection_method_name, measurement_name, overlay_png_buffer, fig_name in overlay_plot_png_tuples:
        truncated_fig_name = fig_name.split("_preprocessed")[0]
        overlay_png = ImageFile(overlay_png_buffer, name=truncated_fig_name)
        plotModel = OverlayPlotModel(analysis=wrapper, user=wrapper.user,
                                     name="Overlay {}".format(peak_detection_method_name)[:50],
                                     based_on_peak_detection_method_name=peak_detection_method_name,
                                     based_on_measurement=measurement_name,
                                     figure=overlay_png)
        plotModel.save()
        overlay_plot_ids.append(plotModel.pk)
    # return cluster_plot_ids, overlay_plot_ids
    print("Finished overlay plots")
    return analysis_id


@app.task
def heatmap_plot(processed_file_id, analysis_id, user_id):
    """
    Make HeatmapPlot and save in backend.
    :param processed_file_id:
    :param analysis_id:
    :param user_id:
    :return:
    """
    from breathpy.model.BreathCore import construct_default_parameters, MccImsMeasurement
    from breathpy.view.BreathVisualizations import HeatmapPlot
    from django.contrib.auth.models import User

    # start analysis process and serve up some pdfs
    plot_params, file_params = construct_default_parameters(file_prefix="", folder_name="", make_plots=True)
    plot_params['use_buffer'] = True

    pf = ProcessedFile.objects.get(pk=processed_file_id)

    # create some intensity matrices
    # heatmap_png = ImageFile(HeatmapPlot.plot_heatmap_helper(
    heatmap_png = ImageFile(HeatmapPlot.FastIntensityMatrix(
        mcc_ims_measurement=MccImsMeasurement.import_from_csv(pf.file.path),
        plot_parameters=plot_params,
    ),
        # is name mis-formated?
        name="heatmap_{}.png".format(pf.name))

    plotModel = IntensityPlotModel(analysis=MccImsAnalysisWrapper.objects.get(pk=analysis_id),
                                   user=User.objects.get(pk=user_id),
                                   name="Heatmap {}".format(pf.name),
                                   based_on_measurement=pf.name,
                                   based_on_peak_detection_method_name='',
                                   figure=heatmap_png)
    plotModel.save()
    return plotModel.pk


@app.task
def classwise_heatmap_plots(analysis_id, user_id):
    """
    Make ClasswiseHeatmapPlot - and save in backend. Use class_label_id_mapping: dict mapping class label to list of
        ids to merge in the plot - the first id will be used to align all DataFrames to
    :param analysis_id: ID of the Analysis this plot belongs to
    :param user_id: owners id
    :return: primary keys of PlotModel Instances created for each class_label
    """
    from breathpy.model.BreathCore import construct_default_parameters, MccImsMeasurement
    from breathpy.view.BreathVisualizations import HeatmapPlot
    from django.contrib.auth.models import User

    # start analysis process and serve up some pdfs
    plot_params, file_params = construct_default_parameters(file_prefix="", folder_name="", make_plots=True)
    plot_params['use_buffer'] = True

    plot_model_ids = []
    class_label_id_mapping = MccImsAnalysisWrapper.objects.get(pk=analysis_id).ims_set.file_set.class_label_processed_id_dict
    for class_label, processed_file_ids in class_label_id_mapping.items():
        # should be label :[ids] not the orther way round
        processed_files = ProcessedFile.objects.filter(pk__in=processed_file_ids)
        processed_measurements = [MccImsMeasurement.import_from_csv(pf.file.path, class_label=class_label) for pf in processed_files]

        _, figure = HeatmapPlot._plot_classwise_heatmap_helper(mccImsMeasurements=processed_measurements, class_label=class_label, plot_parameters=plot_params)

        name = "classwise_heatmap_{}.png".format(class_label)
        heatmap_png = ImageFile(figure, name=name)

        plotModel = ClasswiseHeatMapPlotModel(analysis=MccImsAnalysisWrapper.objects.get(pk=analysis_id),
                                              user=User.objects.get(pk=user_id),
                                              name="Classwise Heatmap {}".format(name),
                                              based_on_measurement_list=processed_file_ids,
                                              based_on_peak_detection_method_name='',
                                              figure=heatmap_png,
                                              class_label=class_label)
        plotModel.save()
        plot_model_ids.append(plotModel.pk)
    return plot_model_ids

@app.task
def process_gcms_measurement(raw_file_id, preprocessing_options):
    """
    Apply either raw - wavelet preprocessing and peak detection or centroided peak detection based on user selection,
        create model instances and save to db.
        Denoising and PeakDetection is applied here
    :param raw_file_id:
    :param preprocessing_options:
    :return:
    """
    import os
    from pathlib import Path
    from breathpy.model.BreathCore import GCMSAnalysis
    from .models import GCMSRawMeasurement, GCMSUnlinkedPeakDetectionResult

    raw_file = GCMSRawMeasurement.objects.get(pk=raw_file_id)
    measurement_fn = raw_file.file.file.name  # absolute path

    print(f"Processing {Path(measurement_fn)}")
    matched_steps = GCMSAnalysis.match_processing_options(preprocessing_options)

    # create GCMSPeakDetectionResult for each peak detection method
    pdr_dict = GCMSAnalysis.detect_peaks_helper(
            input_filename=measurement_fn,
            peak_detection_method=matched_steps['peak_detection_steps'][0],  # only applying one of the pdms - so no loop required
            preprocessing_parameter_dict=GCMSAnalysis.prepare_preprocessing_parameter_dict(preprocessing_options)
        )

    # create unlinked result
    long_result_filename, pdm_name = pdr_dict['feature_storage'], pdr_dict['pdm_name']
    # filename should be short - not absolute path
    result_path = Path(long_result_filename)
    result_filename = str(result_path.stem) + str(result_path.suffix)

    # use buffer to create content file and cleanup the original created file
    with open(result_path, 'rb') as fh:
        with ContentFile(fh.read(), name=result_filename) as buffer:

            updr = GCMSUnlinkedPeakDetectionResult(
                filename=result_filename,
                peak_detection_method_name=pdm_name,
                # raw_measurement = raw_file_id,  # usage for linked result
                # use buffer
                file=buffer)
    updr.save()
    os.remove(fh.name)
    return updr.pk




class GCMSParallelPreprocessingTask(JobtasticTask):
    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1

    def calculate_result(self, analysis_id, **kwargs):

        t0 = time.time()

        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        self.update_progress(1, 100)

        raw_file_ids = wrapper.gcms_set.raw_files

        number_of_measurements = len(raw_file_ids)
        # 1/1 peak detection, no plots
        total_progress = number_of_measurements

        self.update_progress(1, total_progress)

        preprocessing_options = wrapper.preprocessing_options

        # use group to preprocess -  need all peak detection results

        # starts the parallel processing task and returns the raw_ids and pdr_ids
        prep_res = group(process_gcms_measurement.s(raw_file_id, preprocessing_options) for raw_file_id in raw_file_ids)()

        # dangerous - in case we have too many ParallelPreprocessing tasks started at once - we will wait forever - so we fail after 2 hours of preprocessing
        # and important to run celery with -Ofair option - workers kept stalling
        while not prep_res.successful():

            self.update_progress(prep_res.completed_count(), total_progress)
            time.sleep(1)

            if (time.time() - t0) > 120*60:
                raise ValueError("Running for over 2 hours in pre-processing. Something is off.")

        self.update_progress(number_of_measurements, total_progress)
        pdr_ids = prep_res.join()

        # print(pdr_ids)
        # update gcms_fileset
        processing_options = GCMSAnalysis.match_processing_options(wrapper.preprocessing_options)
        peak_detection_method = processing_options['peak_detection_steps'][0]

        name = ""
        gcms_pd_fileset = GCMSPeakDetectionFileSet(name=name, description=name,
                                                   class_label_processed_id_dict=OrderedDict(),
                                                   class_label_dict = wrapper.class_label_mapping,
                                                   peak_detection_results=pdr_ids,
                                                   peak_detection_method_name=peak_detection_method.name
                                                   )

        # dont think we need this
        # gcms_pd_fileset.class_label_processed_id_dict = class_label_processed_id_dict
        gcms_pd_fileset.save()

        # now add pd_fileset to filest and then save in wrapper
        gcms_set = wrapper.gcms_set
        gcms_set.peak_detection_fileset = gcms_pd_fileset
        gcms_set.save()

        wrapper.gcms_set = gcms_set
        wrapper.save()

        self.update_progress(total_progress, total_progress)
        return analysis_id




class GCMSAnalysisEvaluationTask(JobtasticTask):
    """
    Align featureXMLs, create Feature matrix, do cross validation and feature extraction of PeakDetectionResults
    """
    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1


    def calculate_result(self, analysis_id, **kwargs):
        from collections import Counter
        from .models import WebPredictionModel, BoxPlotModel, RocPlotModel
        from breathpy.model.ProcessingMethods import FeatureReductionMethod, PerformanceMeasure
        from breathpy.view.BreathVisualizations import RocCurvePlot, BoxPlot, TreePlot
        import numpy as np


        self.update_progress(1, 100)

        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        # number_of_measurements = len(wrapper.gcms_set.peak_detection_fileset.peak_detection_results)
        number_of_measurements = len(wrapper.class_label_mapping)
        total_progress = 4*number_of_measurements
        self.update_progress(1, total_progress)

        gcms_analysis, plot_params, file_params = wrapper.reinitialize_gcms_analysis()

        processing_options = gcms_analysis.match_processing_options(wrapper.preprocessing_options)
        peak_detection_method = processing_options['peak_detection_steps'][0]
        peak_alignment_method = processing_options['peak_alignment_step']

        # alignment is slow - will take ~20 minutes for algae
        trainings_matrix = gcms_analysis.align_peaks(peak_alignment_method, save_to_disk=False)

        self.update_progress(3, total_progress)
        # now init a mcc_ims_analysis from feature_matrix
        #   create feature matrix model instance for db?

        fm_fn = "{}_feature_matrix".format(peak_detection_method.name)
        buff = StringIO()
        trainings_matrix.to_csv(buff, index=True, header=True, index_label="index")
        buff.seek(0)

        fm = FeatureMatrix(analysis=wrapper,
                           name=fm_fn,
                           peak_detection_method_name=peak_detection_method.name,
                           class_label_dict=wrapper.class_label_mapping,
                           file=ContentFile(buff.getvalue(), name=fm_fn + ".csv"))
        fm.save()


        # only do plots + roc / p_value analysis

        # only need AnalysisResult and class labels

        analysis, plot_params, file_params, coerced_pdm = wrapper.prepare_custom_fm_approach(fm)

        # extra step for gcms analysis - need to set peak_detection_combined to peak-detection methods as constructor filters them out
        analysis.peak_detection_combined = processing_options['peak_detection_steps']

        if FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES in analysis.AVAILABLE_FEATURE_REDUCTION_METHODS:

            reduced_fm = analysis.analysis_result.remove_redundant_features_fm(
                feature_matrix={coerced_pdm.name: trainings_matrix},
                class_label_dict=analysis.get_class_label_dict(),
                # use parameters from initialization
                **analysis.performance_measure_parameter_dict.get(FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES,
                                                                  {}))
        else:
            reduced_fm = {coerced_pdm.name: trainings_matrix}

        analysis.analysis_result.trainings_matrix = reduced_fm


        # get Performance measures directly from analysis object
        performance_measures = []
        for eo in wrapper.evaluation_options.keys():
            try:
                # we also have feature reduction options in evaluation_options
                performance_measures.append(PerformanceMeasure(eo))
            except ValueError:
                pass

        # pycharm confuses the two evaluate_performance functions - will use the one in BreathCore.MccImsAnalysis
        analysis.evaluate_performance(exising_analysis_result=analysis.analysis_result)

        # only export trainings_matrix of best_model
        best_model_pdmname, feature_names, decision_tree_buffer = analysis.get_best_model()

        # update the analysis to only include best option
        preprocessing_options = wrapper.preprocessing_options
        # need to remove all other peak_detection_methods on initialization
        all_available_options = list(analysis.AVAILABLE_PEAK_DETECTION_METHODS)
        all_available_options.extend(analysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS)
        all_available_option_names = [ao.name for ao in all_available_options]

        keys_to_pop = set(all_available_option_names).difference(set([best_model_pdmname]))
        for ktp in keys_to_pop:
            preprocessing_options.pop(ktp, None)
        wrapper.preprocessing_options = preprocessing_options
        wrapper.automatic_selected_method_name = best_model_pdmname
        wrapper.save()
        print(f"Updated preprocessing options to {preprocessing_options}")

        # get number of measurements from smallest class - need at least 2 measurements for cross validation
        class_counts = Counter([class_label for m_name, class_label in analysis.get_class_label_dict().items()])
        min_occurence = min(class_counts.values())
        can_cross_validate = min_occurence >= 10

        # cross validation is a given
        # roc plots can only be done if we applied cross validation
        if can_cross_validate:
            if len(set(analysis.analysis_result.class_labels)) == 2:
                roc_plot_tuples = RocCurvePlot.ROCCurve(
                    analysis.analysis_result, plot_parameters=plot_params,
                    limit_to_peak_detection_method_name=best_model_pdmname)
            else:
                roc_plot_tuples = RocCurvePlot.MultiClassROCCurve(analysis.analysis_result,
                                                                  plot_parameters=plot_params,
                                                                  limit_to_peak_detection_method_name=best_model_pdmname)

            for evaluation_method_name, peak_detection_method_name, roc_plot_buffer, fig_name in roc_plot_tuples:
                roc_png = ImageFile(roc_plot_buffer, name=fig_name)
                plotModel = RocPlotModel(analysis=wrapper, user=wrapper.user,
                                         name="Roc {}".format(peak_detection_method_name),
                                         based_on_peak_detection_method_name=peak_detection_method_name,
                                         based_on_performance_measure_name=evaluation_method_name,
                                         figure=roc_png)
                plotModel.save()

        # boxplots are only created if we have corrected pvalues
        # advanced boxplots here
        box_plot_tuples = BoxPlot.BoxPlotBestFeature(analysis.analysis_result, plot_parameters=plot_params,
                                                     limit_to_peak_detection_method_name=best_model_pdmname)
        for (performance_measure_method_name, peak_detection_method_name, model_of_class,
             peak_id), box_plot_buffer, fig_name in box_plot_tuples:
            box_png = ImageFile(box_plot_buffer, name=fig_name)
            plotModel = BoxPlotModel(analysis=wrapper, user=wrapper.user,
                                     name="BoxPlot {}".format(peak_detection_method_name),
                                     figure=box_png,
                                     based_on_peak_detection_method_name=peak_detection_method_name,
                                     based_on_performance_measure_name=performance_measure_method_name,
                                     based_on_peak_id=peak_id,
                                     )
            plotModel.save()  #


        # render decision trees
        tree_plot_tuples = TreePlot.DecisionTrees(analysis.analysis_result, plot_parameters=plot_params,
                                                  limit_to_peak_detection_method_name=best_model_pdmname)
        for plot_tuple in tree_plot_tuples:
            for (eval_method_name, pdm_name, class_comparison_str), tree_plot_buffer, fig_name in plot_tuple:
                # print(eval_method_name, pdm_name, class_comparison_str, fig_name)
                dt_png = ImageFile(tree_plot_buffer, name=fig_name)
                plotModel = DecisionTreePlotModel(
                    analysis=wrapper, user=wrapper.user,
                    name="DecisionTree {} {}".format(pdm_name, eval_method_name), figure=dt_png,
                    based_on_peak_detection_method_name=pdm_name,
                    based_on_performance_measure_name=eval_method_name,
                )
                plotModel.save()  #

        # prebuild empty stats_dict for FDR - stats are only computed in random forest and cross validation
        # mcc_ims_analysis.analysis_result.pvalues_df

        for eval_method in performance_measures:
            if isinstance(eval_method, str):
                eval_method_name = eval_method
            elif isinstance(eval_method, PerformanceMeasure):
                eval_method_name = eval_method.name
            else:
                eval_method_name = str(eval_method)


            if eval_method_name == "DECISION_TREE_TRAINING":
                # don't do anything - just used to pass along stuff, not actually measure performance
                pass
            else:
                stats_dict = {}
                if eval_method_name != PerformanceMeasure.FDR_CORRECTED_P_VALUE.name:
                    stats_dict = \
                        analysis.analysis_result.analysis_statistics_per_evaluation_method_and_peak_detection[
                            eval_method_name]

                buffer = StringIO()
                # get best_features from analysis result, grouped by eval_method
                best_features_df = analysis.analysis_result.best_features_df.loc[
                    analysis.analysis_result.best_features_df['performance_measure_name'] == eval_method_name]
                best_features_df.to_csv(buffer, index=True, header=True, index_label="index")
                buffer.seek(0)
                stats_model = StatisticsModel(
                    evaluation_method_name=eval_method_name,
                    analysis=wrapper,
                    statistics_dict=stats_dict,
                    best_features_df=ContentFile(buffer.getvalue(), name="best_features_df_{}.csv".format(eval_method_name))
                )
                buffer.close()
                stats_model.save()
                # print(stats_model.best_features_df)

        self.update_progress(4, total_progress)

        # export predictor
        # predictors = mcc_ims_analysis.analysis_result.export_prediction_models(path_to_save="", use_buffer=True)
        # print(predictor)
        class_labels = np.unique(list(analysis.class_label_dict.values())).tolist()

        # create and save prediction model to db
        # print(mcc_ims_analysis.analysis_result.feature_names_by_pdm)
        # from .models import SingleWebPredictionModel
        web_prediction_model = WebPredictionModel(
            name="PredictionModel {} {}".format(analysis.dataset_name, best_model_pdmname),
            scipy_predictor_pickle=ContentFile(decision_tree_buffer,
                                               name="predictor_pickle_{}_{}".format(analysis.dataset_name,
                                                                                    best_model_pdmname)),
            feature_names_by_pdm={best_model_pdmname: feature_names},
            mcc_ims_analysis=wrapper,
            class_labels=class_labels)
        web_prediction_model.save()
        # return web_prediction_model.pk
        self.update_progress(total_progress, total_progress)

        return analysis_id

class GCMSPredictClassTask(JobtasticTask):
    """
    This Task uses a prediction model and predicts the classes of gcms-pdr
    The prediction model requires a realignment of previous pdr - to generate the same peak_ids -
      to use the same columns from model training
    """

    significant_kwargs = [('analysis_id', str)]
    herd_avoidance_timeout = -1
    cache_duration = -1


    def calculate_result(self, analysis_id, gcms_fileset_id=0, feature_matrix_id=0, **kwargs):
        from .models import WebPredictionModel, PredictionResult, FeatureMatrix
        from breathpy.model.BreathCore import PredictionModel
        import joblib
        total_progress = 10

        # get evaluation params from analysis
        wrapper = MccImsAnalysisWrapper.objects.get(pk=analysis_id)

        web_prediction_model = WebPredictionModel.objects.get(mcc_ims_analysis=analysis_id)
        if not any([gcms_fileset_id, feature_matrix_id]):
            raise ValueError("Neither gcms_fileset_id or feature_matrix_id are set.")

        # use preprocess measurement method
        # give peak detection results to prediction
        self.update_progress(1, total_progress)

        # distinguish between feature xml and feature matrix
        using_feature_matrix = False

        if feature_matrix_id:
            feature_matrix = FeatureMatrix.objects.get(pk=feature_matrix_id)
            using_feature_matrix = True

            # this should work easily - maybe pdm /combined need to be set
            mcc_ims_analysis, plot_parameters, file_parameters, coerced_pdm = wrapper.prepare_custom_fm_approach(
                feature_matrix_model=feature_matrix)
            # no alignment
            mcc_ims_analysis.peak_detection_combined = [coerced_pdm]
            # make sure features match
            df_matching_training = PredictionModel.reconstruct_remove_features(
                mcc_ims_analysis.analysis_result.trainings_matrix, web_prediction_model.feature_names_by_pdm)

            self.update_progress(2, total_progress)

            scipy_predictor_by_pdm = joblib.load(web_prediction_model.scipy_predictor_pickle.path)

            # get original labels
            original_labels = feature_matrix.get_class_label_dict()

        elif gcms_fileset_id:
            # using_feature_matrix = False
            gcms_fileset = GCMSFileSet.objects.get(pk=gcms_fileset_id)


            # TODO distinguish between raw and featurexml case
            is_raw = gcms_fileset.peak_detection_fileset is None
            if is_raw:

                # apply full preprocessing

                t0 = time.time()

                raw_file_ids = gcms_fileset.raw_files

                number_of_measurements = len(raw_file_ids)
                # 1/1 peak detection, no plots
                total_progress = number_of_measurements

                self.update_progress(1, total_progress)

                preprocessing_options = wrapper.preprocessing_options

                # use group to preprocess -  need all peak detection results

                # starts the parallel processing task and returns the raw_ids and pdr_ids
                prep_res = group(
                    process_gcms_measurement.s(raw_file_id, preprocessing_options) for raw_file_id in raw_file_ids)()

                # dangerous - in case we have too many ParallelPreprocessing tasks started at once - we will wait forever - so we fail after 2 hours of preprocessing
                # and important to run celery with -Ofair option - workers kept stalling
                while not prep_res.successful():

                    self.update_progress(prep_res.completed_count(), total_progress)
                    time.sleep(1)

                    if (time.time() - t0) > 120 * 60:
                        raise ValueError("Running for over 2 hours in pre-processing. Something is off.")

                self.update_progress(number_of_measurements, total_progress)
                pdr_ids = prep_res.join()

                # update gcms_fileset
                processing_options = GCMSAnalysis.match_processing_options(wrapper.preprocessing_options)
                peak_detection_method = processing_options['peak_detection_steps'][0]

                name = ""
                gcms_pd_fileset = GCMSPeakDetectionFileSet(name=name, description=name,
                                                           class_label_processed_id_dict=OrderedDict(),
                                                           class_label_dict=gcms_fileset.class_label_processed_id_dict,
                                                           peak_detection_results=pdr_ids,
                                                           peak_detection_method_name=peak_detection_method.name
                                                           )

                # dont think we need this
                # gcms_pd_fileset.class_label_processed_id_dict = class_label_processed_id_dict
                gcms_pd_fileset.save()

                # now add pd_fileset to filest
                gcms_fileset.peak_detection_fileset = gcms_pd_fileset
                gcms_fileset.save()

                # gcms_fileset not associated with wrapper - only with prediction

            # now continue with feature xml stuff
            gcms_analysis, plot_params, file_params = wrapper.reinitialize_gcms_analysis(gcms_fileset_id=gcms_fileset_id)

            # align features and create test_matrix
            # TODO use previous alignment results instead of aligning all of them again - would need additional datastructures

            processing_options = gcms_analysis.match_processing_options(wrapper.preprocessing_options)
            peak_detection_method = processing_options['peak_detection_steps'][0]
            peak_alignment_method = processing_options['peak_alignment_step']

            # alignment is slow - will take ~20 minutes for algae
            full_matrix = gcms_analysis.align_peaks(peak_alignment_method, save_to_disk=False)

            # only really interested in test_matrix
            tr_labels = wrapper.gcms_set.get_class_label_processed_id_dict()
            test_labels = gcms_fileset.get_class_label_processed_id_dict()
            tr_matrix, test_matrix = gcms_analysis.split_matrix_train_test(full_matrix, tr_labels, test_labels)

            #   create feature matrix model instance for db

            fm_fn = "{}_feature_matrix".format(peak_detection_method.name)
            buff = StringIO()
            test_matrix.to_csv(buff, index=True, header=True, index_label="index")
            buff.seek(0)

            fm = FeatureMatrix(analysis=wrapper,
                               name=fm_fn,
                               peak_detection_method_name=peak_detection_method.name,
                               class_label_dict=test_labels,
                               file=ContentFile(buff.getvalue(), name=fm_fn + ".csv"),
                               is_prediction_matrix=True,
                               is_training_matrix=False)
            fm.save()


            mcc_ims_analysis, plot_parameters, file_parameters, coerced_pdm = wrapper.prepare_custom_fm_approach(
                feature_matrix_model=fm)
            mcc_ims_analysis.peak_detection_combined = [peak_detection_method]
            # no alignment

            # make sure features match
            df_matching_training = PredictionModel.reconstruct_remove_features(
                mcc_ims_analysis.analysis_result.trainings_matrix, web_prediction_model.feature_names_by_pdm)


            scipy_predictor_by_pdm = joblib.load(web_prediction_model.scipy_predictor_pickle.path)

            # get original labels
            original_labels = fm.get_class_label_dict()

            # print(f"original_labels: {original_labels}")

            # get labels to assign class and return prediction
            # get evaluation matrix for each chosen peak_detection method

        # rest of processing - both fm - approach, raw and featureXML should have the same parameters set
        prediction_by_pdm = dict()
        for pdm in mcc_ims_analysis.peak_detection_combined:
            X_reduced = df_matching_training[pdm.name]
            # Import exisitng model and predict classes of unknown data
            # probas = self.scipy_predictor.predict_proba(X)
            prediction_matrix = X_reduced.fillna(0)

            buff = StringIO()
            prediction_matrix.to_csv(buff, index=True, header=True, index_label="index")
            buff.seek(0)
            pm_fn = f"prediction_matrix_{pdm.name}.csv"

            # create FeatureMatrix and save in backend for potential export
            prediction_matrix_model = FeatureMatrix(
                analysis=wrapper, name=f"Prediction Matrix {pdm.name}",
                peak_detection_method_name=pdm.name,
                file=ContentFile(buff.getvalue(), name=pm_fn),
                is_training_matrix=False, is_prediction_matrix=True)
            prediction_matrix_model.save()

            prediction_by_pdm[pdm] = scipy_predictor_by_pdm[pdm].predict(prediction_matrix).tolist()

        # print(f"prediction_by_pdm[pdm] {prediction_by_pdm[pdm]}")

        labeled_prediction_result = {}

        for pdm, prediction_result in prediction_by_pdm.items():
            # get measurement_names from feature_matrix index
            measurement_names = df_matching_training[pdm.name].index
            labeled_prediction_result[pdm] = {measurement_name: web_prediction_model.class_labels[class_index] for
                                              measurement_name, class_index in
                                              zip(measurement_names, prediction_result)}
            # print(f"labeled_prediction_result[pdm] {labeled_prediction_result[pdm]}")
            # print(f"prediction_result {prediction_result}")

        for pdm, prediction_result in labeled_prediction_result.items():
            prediction_result_model = PredictionResult(
                web_prediction_model=web_prediction_model,
                class_assignment=prediction_result,
                peak_detection_method_name=pdm.name,
                original_class_labels=original_labels,
            )
            prediction_result_model.save()

        self.update_progress(total_progress, total_progress)

        return web_prediction_model.pk