from collections import Counter, OrderedDict
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.shortcuts import render, redirect, get_object_or_404, reverse, render_to_response, HttpResponse
from django.http import HttpResponseForbidden

from .models import FileSet, WebImsSet, ZipFileValidator

from .views import (temp_or_login_required, if_temp_update_context, pop_non_user_session_keys,
                    create_evaluation_params_from_form, prepare_prediction_template_parameters, PlotRetriever)
from .forms import (GCMSAnalysisForm, GCMSProcessingForm, GCMSProcessingFormMatcher, GCMSEvaluationForm)
from .models import (MccImsAnalysisWrapper, GCMSPeakDetectionFileSet, GCMSFileSet, FeatureMatrix,
                     WebPredictionModel, PredictionResult,
                     RocPlotModel, BoxPlotModel, DecisionTreePlotModel,)

from breathpy.model.ProcessingMethods import (GCMSPeakDetectionMethod)

@csrf_protect
def _selectDatasetGMCS(request, user):
    template_name = 'breath/gcms_selectdataset.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)

    # create gcms based fileset
    if request.method == "POST":
        gcms_analysis_form = GCMSAnalysisForm(request.POST, user_id=user.pk)

    # if GET or other request create default form
    else:
        gcms_analysis_form = GCMSAnalysisForm(user_id=user.pk)
        context['form'] = gcms_analysis_form
        return render(request, template_name, context)

    # management over session and ids
    if gcms_analysis_form.is_valid():
        wrapper = MccImsAnalysisWrapper(
            user=user,
            preprocessing_options=OrderedDict(),
            class_label_mapping=OrderedDict(),
            evaluation_options=OrderedDict(),
            is_gcms_analysis=True,
        )
        wrapper.save()
        analysis_id = wrapper.pk

        # distinguish between peak_deteciton_results, feature matrix and raw_files
        #   raw_files -> need parameter selection -> processing + peak-detection -> Evaluation parameter selection
        #   pdr -> evaluation parameter selection
        #   feature_matrix -> evaluation parameter selection
        feature_matrix_id = gcms_analysis_form.cleaned_data.get('feature_matrix_id', False)
        if feature_matrix_id:
            # forward to evaluation

            # associate fm with analysis so it's not lost
            fm = FeatureMatrix.objects.get(pk=feature_matrix_id)

            wrapper.preprocessing_options = OrderedDict({fm.peak_detection_method_name: {}})
            wrapper.is_custom_feature_matrix_analysis = True
            wrapper.class_label_mapping = fm.class_label_dict
            wrapper.save()

            fm.analysis = wrapper
            fm.save()


            gcms_analysis_type = "feature_matrix"
            on_success_redirect_to = "gcms_evaluation"
            request.session['feature_matrix_id'] = feature_matrix_id
            request.session['gcms_analysis_type'] = gcms_analysis_type

            return redirect(
                reverse(on_success_redirect_to, kwargs={'analysis_id': analysis_id}))

        gcms_fileset_id = gcms_analysis_form.cleaned_data.get('gcms_fileset_id', False)

        gcms_fileset = GCMSFileSet.objects.get(pk=gcms_fileset_id)
        wrapper.gcms_set = gcms_fileset
        # make sure class_label dict is ready from the beginning
        wrapper.class_label_mapping = gcms_fileset.get_class_label_processed_id_dict()
        wrapper.save()

        # both featureXML and raw have fileset created - check for raw_measurements to know which
        if len(gcms_fileset.raw_files):
            # is_raw = True
            gcms_analysis_type = "raw"
            on_success_redirect_to = "select_parameters_gcms"
            request.session['gcms_fileset_id'] = gcms_fileset_id
            request.session['gcms_analysis_type'] = gcms_analysis_type
            #  forward to parameter selection and preprocessing\

            return redirect(
                reverse(on_success_redirect_to, kwargs={'analysis_id': analysis_id}))

        # featurexml case
        else:
            # forward to evaluation
            gcms_analysis_type = "feature_xml"
            on_success_redirect_to = "gcms_evaluation"
            request.session['gcms_fileset_id'] = gcms_fileset_id
            request.session['gcms_analysis_type'] = gcms_analysis_type

            # add peak detection method CUSTOM to preprocessing methods - needed for evaluation
            # update not hardcode - just in case we decide to implement coercion for feature XML
            orig_prepro = wrapper.preprocessing_options
            orig_prepro[gcms_fileset.peak_detection_fileset.peak_detection_method_name] = {}
            wrapper.preprocessing_options = orig_prepro
            wrapper.save()

            return redirect(
                reverse(on_success_redirect_to, kwargs={'analysis_id': analysis_id}))

    # form is invalid
    else:
        context['not_valid'] = "Yes"
        context['form'] = gcms_analysis_form
    return render(request, template_name, context)


@temp_or_login_required
@csrf_exempt
def selectDatasetGCMS(request, user):
    """
    Create analysis object if POST and forward to evaluation / parameter selection based on dataset selection
    Render template if GET
    :param request:
    :param user:
    :return:
    """
    # required due to handling of files in different way when they have different sizes, we always want a temporary file for zips, not an in memory file
    # see https://stackoverflow.com/questions/6906935/problem-accessing-user-uploaded-video-in-temporary-memory
    # and https://docs.djangoproject.com/en/dev/topics/http/file-uploads/#modifying-upload-handlers-on-the-fly
    # removes the first file handler (MemoryFile....)
    # pop all session stuff
    # pop everything that's not user / user_id or starts with _
    pop_non_user_session_keys(request, exceptions=["gcms_fileset_id", "feature_matrix_id"])  # except for gcms_fileset_id and feature_matrix_id
    print(request.session.keys())
    return _selectDatasetGMCS(request, user=user)



@temp_or_login_required
def selectParametersGCMS(request, analysis_id, user):
    """
    If POST create and execute parallel pre-processing task
    If GET return form to select parameters
    :param request:
    :param analysis_id:
    :param user:
    :return:
    """
    from .tasks import GCMSParallelPreprocessingTask
    template_name = 'breath/gcms_selectparameters.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)

    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    # process POST requests
    if request.method == 'POST':
        processing_form = GCMSProcessingForm(request.POST)
        # validate Form
        if processing_form.is_valid():

            processingStepsFormMatcher = GCMSProcessingFormMatcher(processing_form)
            preprocessing_parameters = processingStepsFormMatcher.preprocessing_parameters

            # now convert to jsonable format
            jsonable_preprocessing_parameters = {k.name:v for k, v in preprocessing_parameters.items()}

            # let's hope those are still in the session
            gcms_fileset_id = request.session['gcms_fileset_id']
            # gcms_analysis_type = request.session['gcms_analysis_type']

            # update the wrapper object
            wrapper.gcms_set = GCMSFileSet.objects.get(pk=gcms_fileset_id)
            wrapper.preprocessing_options = jsonable_preprocessing_parameters

            wrapper.save()

            result = GCMSParallelPreprocessingTask.delay_or_fail(analysis_id=analysis_id)

            #   result of task GCMSFileset
            return redirect(
                reverse('gcms_preprocessing_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))
        else:
            context['not_valid'] = "Yes"
            context['form'] = processing_form
            return render(request, template_name, context)

    # if get or other request create default form
    else:
        # Initial values are set in class definition
        processing_form = GCMSProcessingForm()
        context['form'] = processing_form
        return render(request, template_name, context)


@temp_or_login_required
def gcms_evaluation(request, analysis_id, user):
    """
    Allow user to specify evaluation parameters for custom analysis with pdr and fm
    :param request:
    :param analysis_id:
    :param is_using_fm: int
    :param user:
    :return:
    """
    from .tasks import GCMSAnalysisEvaluationTask, CustomPeakDetectionFMAnalysisTask

    template_name = 'breath/gcms_select_evaluation.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)

    # create analysis
    # run featureXML and fm
    # get object via wrapper and analysis_id

    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    is_using_fm = False  # shouldn't reach here - we redirect to different pipeline if fm
    # extract minimum occurence from specific model objects
    # gcms_analysis_type = request.session.get('gcms_analysis_type', "")
    #
    # if gcms_analysis_type == 'feature_xml' or gcms_analysis_type == 'raw':
    #     gcms_fileset_id = request.session['gcms_fileset_id']
    #     gcms_fileset = GCMSFileSet.objects.get(pk=gcms_fileset_id)
    #     class_label_dict = gcms_fileset.get_class_label_processed_id_dict()

    # elif gcms_analysis_type == 'feature_matrix':
    #     feature_matrix_id = request.session['feature_matrix_id']
    #     feature_matrix = FeatureMatrix.objects.filter(analysis_id__eq=analysis_id)


    # else:
    #     raise ValueError("Problem with session variable - missing or invalid `gcms_analysis_type`")

    sorted_occurences = sorted(Counter(wrapper.class_label_mapping.values()).values())
    minimum_occurences_class_label = 0
    if sorted_occurences:
        minimum_occurences_class_label = sorted_occurences[0]

    # process submitted form
    if request.method == "POST":
        evaluation_form = GCMSEvaluationForm(minimum_occurences_class_label, request.POST)

        # validate Form
        if evaluation_form.is_valid():

            # get all parameters from analysis form and update evaluation parameters in WebImsSet
            alignment_option = evaluation_form.cleaned_data['peak_alignment']

            #update preprocessing options with alignment
            preprocessing_options = wrapper.preprocessing_options
            preprocessing_options.update({alignment_option.name: {}})
            evaluation_options = create_evaluation_params_from_form(evaluation_form)

            # fileset has featureXML / peak detection options

            wrapper.preprocessing_options = preprocessing_options
            wrapper.evaluation_options = evaluation_options  # somehow already is string not enum

            wrapper.save()

            # start async task and redirect to polling point
            if wrapper.is_custom_feature_matrix_analysis:
                # create task
                # debug_custom_fm_analysis(custom_analysis.pk)
                # this should already work - not different from other featureMatrix processing
                result = CustomPeakDetectionFMAnalysisTask().delay_or_fail(analysis_id=analysis_id)
            else:
                # create task
                # result = debug_custom_pdr_analysis(analysis_id=custom_analysis.pk)
                result = GCMSAnalysisEvaluationTask().delay_or_fail(analysis_id=analysis_id)

            return redirect(
                reverse('gcms_evaluation_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

        else:
            context['not_valid'] = "Yes"
            context['form'] = evaluation_form

    else:

        # request.method == GET - so we render page with form
        evaluation_form = GCMSEvaluationForm(minimum_occurences_class_label=minimum_occurences_class_label)

        # send form back to same url
        evaluation_form.helper.form_action = reverse('gcms_evaluation', kwargs={'analysis_id': analysis_id})
        context['form'] = evaluation_form

    return render(request, template_name=template_name, context=context)



@temp_or_login_required
def gcms_prediction(request, analysis_id, user):
    from django.core.exceptions import MultipleObjectsReturned
    from .tasks import CustomPredictClassTask, GCMSPredictClassTask
    from .models import WebPredictionModel
    from .forms import GCMSPredictionForm

    # implement with gcms - should be extremely similar to mcc ims
    template_name = 'breath/gcms_prediction.html'
    context = {'active_page': 'prediction',
               }
    print("Reached custom prediction")

    mcc_ims_analysis_wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)
    if not mcc_ims_analysis_wrapper.user == user:
        return HttpResponseForbidden()
    context = if_temp_update_context(user=user, context=context)

    # get prediction model by reference to analysis
    # make sure to handle MultipleObjectsReturned
    try:
        # serve 404 if no WebPredictionModel exists
        # --
        prediction_model = get_object_or_404(WebPredictionModel, mcc_ims_analysis=analysis_id)
    except MultipleObjectsReturned as mor:
        prediction_model = WebPredictionModel.objects.filter(mcc_ims_analysis=analysis_id)[0]
        print(f"Got mulitple Webpredictionresults, defaulting to {prediction_model}")


    # if set, we can continue
    prediction_model_pk = prediction_model.pk
    # request.session['analysis_id'] = analysis_id
    if request.method == "POST":

        gcms_prediction_form = GCMSPredictionForm(prediction_model_pk, request.POST, user_id=user.pk)

        if gcms_prediction_form.is_valid():

            # differentiate when using custom_feature_matrix_analysis
            using_feature_matrix = gcms_prediction_form.cleaned_data.get('feature_matrix_id', False)

            if using_feature_matrix:
                feature_matrix_id = using_feature_matrix
                # custom_feature_matrix = get_object_or_404(FeatureMatrix, pk=feature_matrix_id)
                # use the GCMSPredictClassTask
                result = GCMSPredictClassTask().delay_or_fail(analysis_id=analysis_id, feature_matrix_id=feature_matrix_id)

            else:
                gcms_fileset_id = gcms_prediction_form.cleaned_data['gcms_fileset_id']
                result = GCMSPredictClassTask().delay_or_fail(analysis_id=analysis_id, gcms_fileset_id=gcms_fileset_id)

            return redirect(
                reverse('gcms_prediction_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

    # GET - we need to initialize the form
    else:
        gcms_prediction_form = GCMSPredictionForm(web_prediction_model_key=prediction_model_pk, user_id=user.pk)

    # gcms specific preparation?
    context = prepare_prediction_template_parameters(context=context, analysis_id=analysis_id, user=user, use_custom_prediction=True)
    context['form'] = gcms_prediction_form
    return render(request, template_name, context=context)


@temp_or_login_required
def gcms_prediction_result(request, analysis_id, user):

    template_name = 'breath/gcms_prediction_result.html'
    context = {'active_page': 'prediction',
               }

    # get prediction model matching analysis from DB
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    web_prediction_model = get_object_or_404(WebPredictionModel, mcc_ims_analysis=wrapper)

    if not web_prediction_model.mcc_ims_analysis.user == user:
        return HttpResponseForbidden()
    context = if_temp_update_context(user=user, context=context)

    # get all results associated with web_prediciton model
    prediction_results = PredictionResult.objects.all().filter(web_prediction_model=web_prediction_model.pk)

    result_dicts = []
    for pr in prediction_results:
        orig_labels = {}
        if pr.original_class_labels:
            orig_labels = pr.original_class_labels
        result_dicts.append({
            # add orig_label default "" to match new table style
            "class_assignment": [(m_name, label, orig_labels.get(m_name, "")) for m_name, label in pr.class_assignment.items()],
            "peak_detection_method_name": pr.peak_detection_method_name,
            "created_at": pr.created_at.strftime('%Y-%m-%d %H:%M'),
            "id": pr.pk,
        })

    context['prediction_results'] = result_dicts
    pr = PlotRetriever(analysis_id)

    # if automatic analysis we select a best_model automatically - so only get the plots that match the eak_detection_method name
    images, available_tags = pr.get_plots_of_analysis(
        [RocPlotModel, BoxPlotModel, DecisionTreePlotModel])
    context['images'] = images
    context['available_tags'] = available_tags
    context['analysis_id'] = analysis_id
    return render(request, template_name, context=context)


@temp_or_login_required
def gcms_preprocessing_progress(request, task_id, analysis_id, user):
    # check if user has access to check the status, or if we shared the task_id
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    gcms_evaluation_url = reverse('gcms_evaluation', kwargs={'analysis_id': analysis_id})
    print(gcms_evaluation_url)

    # "celery-tasks"
    task_url = reverse('task_status', kwargs={'task_id': task_id})
    context = {'task_id': task_id, 'result_url': gcms_evaluation_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, template_name='breath/display_progress.html', context=context)


@temp_or_login_required
def gcms_evaluation_progress(request, task_id, analysis_id, user):
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    prediction_url = reverse('gcms_prediction', kwargs={'analysis_id': analysis_id})
    # print(prediction_result_url)
    task_url = reverse('task_status', kwargs={'task_id': task_id})

    context = {'task_id': task_id, 'result_url': prediction_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, 'breath/display_progress.html', context=context)


@temp_or_login_required
def gcms_prediction_progress(request, task_id, analysis_id, user):
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    prediction_result_url = reverse('gcms_prediction_result', kwargs={'analysis_id': analysis_id})
    # print(prediction_result_url)
    task_url = reverse('task_status', kwargs={'task_id': task_id})

    context = {'task_id': task_id, 'result_url': prediction_result_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, 'breath/display_progress.html', context=context)