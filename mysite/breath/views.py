from collections import Counter
from pathlib import Path
import numpy as np
from io import StringIO, BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

from django.shortcuts import render, redirect, get_object_or_404, reverse, render_to_response, HttpResponse
from django.core.exceptions import FieldDoesNotExist
from django.conf import settings
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt, csrf_protect

from breath.cleanup_db import clean_up

from breathpy.model.BreathCore import (
    construct_default_parameters,
    MccImsAnalysis, GCMSAnalysis
                 )

from breathpy.model.ProcessingMethods import (
    PeakDetectionMethod, GCMSPeakDetectionMethod, PeakAlignmentMethod,
    ExternalPeakDetectionMethod,
    PerformanceMeasure,
)

from .decorators import temp_or_login_required

from .forms import (ProcessingStepsForm,
                    ProcessingStepsFormMatcher,
                    WebImsSetForm,
                    CrispyReviewForm,
                    PredictionForm,
                    CustomPredictionForm,
                    AnalysisForm,
                    AnalysisFormMatcher,
                    CustomDetectionAnalysisForm,
                    SignUpForm,
                    UploadUserDatasetForm,
                    )
from .models import (
                    TempUser, TempUserManager,
                    MccImsAnalysisWrapper, WebImsSet, FileSet, WebCustomSet,
                    HeatmapPlotModel, IntensityPlotModel, OverlayPlotModel, ClusterPlotModel, BestFeaturesOverlayPlot, ClasswiseHeatMapPlotModel,
                    RocPlotModel, BoxPlotModel, FeatureMatrix,
                    WebPredictionModel, ClassPredictionFileSet, PredictionResult, StatisticsModel, DecisionTreePlotModel,
                    AnalysisType, UserDefinedFileset, UserDefinedFeatureMatrix, PredefinedFileset, PredefinedCustomPeakDetectionFileSet,
                    GCMSPredefinedPeakDetectionFileSet,
                    construct_user_defined_feature_matrices_from_zip, construct_user_defined_fileset_from_zip,
                    )

def update_user_context_from_request(request, context):
    """
    Show user is logged on pages that do not require login - hack for temp_user
    :param request:
    :param context:
    :return:
    """
    from .models import User
    user = False
    user_id = request.session.get('user_id', '')
    if user_id:
        user = User.objects.get(pk=user_id)
    if login_required(request).user.is_authenticated:
        user = request.user
    if user:
        context = if_temp_update_context(user=user, context=context)
    return context

@temp_or_login_required
def index(request, user):
    template_name = 'breath/index.html'
    context = {'active_page': 'home',
               }
    context = update_user_context_from_request(request, context)

    clean_up(30)
    return render(request, template_name, context=context)

def lazy_logout(request):
    request.session.clear()
    return redirect(reverse("logout"))

def if_temp_update_context(user, context):
    """
    Add is_temp flag to context for templates - is_temp_user or registered
    :param user:
    :param context:
    :return:
    """
    is_temp = not user.email# or ''
    context['is_temp'] = is_temp
    return context


# from https://b0uh.github.io/protect-django-media-files-per-user-basis-with-nginx.html
@temp_or_login_required
def media_access(request, path, user):
    """
     When trying to access :
     myproject.com/media/uploads/passport.png

     If access is authorized, the request will be redirected to
     myproject.com/protected/media/uploads/passport.png

     This special URL will be handle by nginx with the help of X-Accel --> only works when serving with nginx - eg production
     """
    access_granted = False
    # user = request.user
    session_user_id = request.session.get('user_id', '')
    is_tmp_user = request.session.get('tmp_user_id', '')

    # if we logged out - user_id will not be set - unless set in @temp_or_login_required
    if (user.is_authenticated or is_tmp_user) and session_user_id:
        if user.is_staff:
            # If admin, everything is granted
            access_granted = True
        else:
            # For simple user, only their documents can be accessed
            # we check path instead - we save them in user directory
            # due to manual url handling we are slightly vulnerable to url tampering
            # path should be user_<id>/figures/more_path
            path_prefix = path.split("/")[0]
            str_usr_id = path_prefix.split("user_")[1]
            usr_id = int(str_usr_id)
            if session_user_id == usr_id:
            # id in session and figure matches -> save to yield
                access_granted = True
            else:
                print(f"mismatch {session_user_id} == {usr_id}")
    if access_granted:
        response = HttpResponse()
        # Content-type will be detected by nginx
        del response['Content-Type']
        response['X-Accel-Redirect'] = '/protected/media/' + path
        return response
    else:
        return HttpResponseForbidden('You are not authorized to access this media.')


def signup(request):
    template_name = 'registration/signup.html'
    if request.user.is_active:
        # already rgistered users shouldn't register again
        return redirect(reverse('index'))
    tmp_user_id = request.session.get('tmp_user_id', '')
    # is_temp = not request.user.email

    # check that user_id is the same before and after signup
    print(f"User_id before signup {request.session.get('user_id','')}")
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            if tmp_user_id:
                # do merging with tempuser
                # temp_user = TempUser.objects.get(pk=tmp_user_id)
                tmpUserManager = TempUserManager()
                new_user = tmpUserManager.convert(form, temp_user_id=tmp_user_id)
                # temp_user, tmp_username = tmpUserManager.create_temp_user()
            else:
                form.save()
                username = form.cleaned_data.get('username')
                raw_password = form.cleaned_data.get('password1')
                new_user = authenticate(username=username, password=raw_password)
                # new_user.save()
            login(request, new_user)
            request.session['user_id'] = new_user.pk

            if tmp_user_id:
                request.session.pop('tmp_user_id')
            print(f"User_id after signup {request.session.get('user_id', '')}")
            return redirect('index')
    else:
        form = SignUpForm()
    return render(request, template_name, {'form': form})

# TODO users should be able to delete their account - rely on admin atm
# @login_required
# def user(request, user_id):
#     template_name = 'breath/user.html'
#     context = {
#         'active_page': 'user',
#         'user_id': user_id,
#     }
#
#     return render(request, template_name, context=context)


@temp_or_login_required
def list_datasets(request, user):
    # remove old database entries
    clean_up(30)

    template_name = 'breath/list_datasets.html'
    context = {'active_page': 'datasets',
               }
    context = if_temp_update_context(user=user, context=context)

    # TODO prep filesets for context
    # date format:  {self.created_at.strftime('%Y.%m.%d %H:%M')}
    user_fms = UserDefinedFeatureMatrix.objects.filter(user=user).order_by('created_at').reverse()
    user_fss = UserDefinedFileset.objects.filter(user=user).order_by('uploaded_at').reverse()
    # TODO sort by date - intermed object

    default_fss = PredefinedFileset.objects.all()
    default_pd_fs = PredefinedCustomPeakDetectionFileSet.objects.all()
    default_fxml_fs = GCMSPredefinedPeakDetectionFileSet.objects.all()

    # FIXME extend with available GCMSPredefinedPeakDetectionFileSet

    context['user_fms'] = user_fms
    context['user_fss'] = user_fss
    context['default_fss'] = default_fss
    context['default_pd_fs'] = default_pd_fs
    context['default_fxml_fs'] = default_fxml_fs

    return render(request, template_name, context)


@temp_or_login_required
def upload_dataset(request, user):
    # remove old database entries
    clean_up(30)

    template_name = 'breath/upload_dataset.html'
    context = {'active_page': 'datasets',
               }
    context = if_temp_update_context(user=user, context=context)

    if request.method == "POST" and request.FILES:
        upload_form = UploadUserDatasetForm(request.POST, request.FILES)
    # if GET or other request create default form
    else:
        upload_form = UploadUserDatasetForm()
        context['form'] = upload_form
        return render(request, template_name, context)
    if upload_form.is_valid():
        # print(upload_form.cleaned_data.get('zip_file_path'))
        # model needs user - so is created in view instead of in form
        # distinguish on user selection
        analysis_type = upload_form.cleaned_data['analysis_type']
        if analysis_type == AnalysisType.FEATURE_MATRIX.name:
            construct_user_defined_feature_matrices_from_zip(
                zippath=upload_form.cleaned_data['zip_file_path'], user=user, train_val_ratio=upload_form.cleaned_data['train_validation_ratio'],
                name=upload_form.cleaned_data['name'], description=upload_form.cleaned_data['description'],
            )
        elif analysis_type == AnalysisType.RAW_MCC_IMS.name or analysis_type == AnalysisType.RAW_MZML.name:
            construct_user_defined_fileset_from_zip(
                zippath=upload_form.cleaned_data['zip_file_path'], user=user, train_val_ratio=upload_form.cleaned_data['train_validation_ratio'],
                name=upload_form.cleaned_data['name'], description=upload_form.cleaned_data['description'],
                analysis_type=upload_form.cleaned_data['analysis_type'],
            )
        else:
            print("Invalid comparison. Skipping set creation.")
        # upload_form.cleaned_data['zip_file_path'] - is temporary filepath - is cleared when tempdir or InMemory
        return redirect('list_datasets')
    else:
        context['form'] = upload_form
    return render(request, template_name, context)


@temp_or_login_required
def run(request, user):
    # remove old database entries
    clean_up(30)

    # select between mcc/ims and gc/ms pipeline
    template_name = 'breath/run.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)
    return render(request, template_name, context)


@temp_or_login_required
def run_mcc(request, user):
    # select between automatic and custom processing pipeline
    template_name = 'breath/run_mcc.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)
    return render(request, template_name, context)


@temp_or_login_required
@csrf_exempt
def selectDatasetAuto(request, user):
    # ex: /breath/selectdatasetauto/ --> reviewauto
    # path('selectdatasetauto', views.selectDatasetAuto, name='selectDatasetAuto'),

    # required due to handling of files in different way when they have different sizes, we always want a temporary file for zips, not an in memory file
    # see https://stackoverflow.com/questions/6906935/problem-accessing-user-uploaded-video-in-temporary-memory
    # and https://docs.djangoproject.com/en/dev/topics/http/file-uploads/#modifying-upload-handlers-on-the-fly
    # removes the first file handler (MemoryFile....)
    # pop all session stuff
    # dont rely on analysis_id in session - makes multi-analysis impossible
    pop_non_user_session_keys(request)
    return _selectDataset(request, user=user, template_name='breath/selectdatasetauto.html', on_success_redirect_to='reviewAuto')


@temp_or_login_required
@csrf_protect
def reviewAuto(request, user):
    # ex: /breath/reviewauto/  --> progressbar --> prediction
    # path('reviewauto', views.reviewAuto, name='reviewAuto'),
    from .forms import ReviewAutoForm
    from .tasks import ParallelAutomaticPreprocessingEvaluationTask

    template_name = 'breath/review_details_auto.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)
    ims_set = get_object_or_404(WebImsSet, pk=request.session['dataset_info']['ims_set_pk'])
    # run_params = create_preprocessing_params_from_session_forms(request.session['dataset_info'])
    # enable all preprocessing options on default by using default Form
    # update ims_analysis with preprocessing options and evaluation options before submitting and display them

    if request.method == "POST":
        # only required temporary session storage of analysis_id - remove afterwards
        analysis_id = request.session['analysis_id']
        request.session.pop('analysis_id')
        review_form = ReviewAutoForm(request.POST)
        if 'continue' in request.POST:
            # start full process
            template_name = 'breath/run.html'

            # Preprocess the dataset
            # Evaluate the dataset
            # Rank models by performance

            request.session['start_preprocessing'] = True
            # mcc_ims_wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id)
            # request.session.pop('start_preprocessing')
            dataset_info = request.session.get('dataset_info', {})
            if dataset_info:
                result = ParallelAutomaticPreprocessingEvaluationTask.delay_or_fail(analysis_id=analysis_id)

                return redirect(
                    reverse('automatic_analysis_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

        elif 'cancel' in request.POST:
            # return to previous view --> selectDatasetAuto
            return redirect(selectDatasetAuto)
    else:
        review_form = ReviewAutoForm()
        analysis_id = MccImsAnalysisWrapper.create_automatic_analysis(ims_set, user)
        request.session['analysis_id'] = analysis_id
        analysis_wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)
        context.update(prepare_analysis_details_for_context(analysis_wrapper, add_images=False))

    context['review_form'] = review_form
    return render(request, template_name, context)

@temp_or_login_required
def analysis_list(request, user):
    """
    show all analysis ids for user
    images for analysis
    prediction Result
    chosen preprocessing options
    Download options of original / intermediate data
    remove old database entries

    :param request:
    :param user:
    :return:
    """
    clean_up(30)

    template_name = 'breath/analysis_list.html'
    context = {'active_page': 'results',
               }
    context = if_temp_update_context(user=user, context=context)
    qs = MccImsAnalysisWrapper.objects.filter(user=user).order_by('pk')
    analysis_ids = [analysis.pk for analysis in qs]

    context['analysis_ids'] = analysis_ids

    return render(request, template_name, context=context)

@temp_or_login_required
def analysis_details(request, analysis_id, user):
    """
    Creates context for chosen preprocessing options, images, prediction Result and download options of original / intermediate data (FM)
    :param request:
    :param analysis_id:
    :param user:
    :return:
    """
    clean_up(30)
    template_name = 'breath/analysis_details.html'
    context = {'active_page': 'results',
               }
    context = if_temp_update_context(user=user, context=context)
    # check if user has access
    analysis_wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)
    context.update(prepare_analysis_details_for_context(analysis_wrapper))
    plot_retriever = PlotRetriever(analysis_id)
    plot_model_instances = [ClusterPlotModel, ClasswiseHeatMapPlotModel, BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel,
         DecisionTreePlotModel]
    context['images'], context['available_tags'] = plot_retriever.get_plots_of_analysis(model_instance_list=plot_model_instances)
    return render(request, template_name, context=context)

def prepare_analysis_details_for_context(mccImsAnalysisWrapper, add_images=False):
    """
    return dict that is json serializable and holds info that can be used in context to display Analysis Details
    :param mccImsAnalysisWrapper:
    :param add_images: add images to context - used for review endpoints
    :return: dict()
    """
    context = {}
    analysis_id = mccImsAnalysisWrapper.pk
    context['analysis_pk'] = analysis_id

    if add_images and mccImsAnalysisWrapper:
        # check if user has access
        pr = PlotRetriever(analysis_id)
        images = pr.get_plots_of_analysis(
            [IntensityPlotModel, OverlayPlotModel, ClusterPlotModel, BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel,
             DecisionTreePlotModel])
        # print(images)
        # add all matching images to context
        context['images'] = images

    # dataset_info_dict = mccImsAnalysisWrapper.ims_set.get_dataset_info()
    dataset_info_dict = mccImsAnalysisWrapper.get_dataset_info()
    # trainings_matrices included in dataset_info - using pk, pdm and display_feature_matrix
    context['trainings_matrices'] = dataset_info_dict['trainings_matrices']
    context['reduced_trainings_matrices'] = dataset_info_dict['reduced_trainings_matrices']
    context['prediction_matrices'] = dataset_info_dict['prediction_matrices']

    # distinguish between gcms and mccims
    if mccImsAnalysisWrapper.is_gcms_analysis:
        preprocessing_steps_dict = GCMSAnalysis.match_processing_options(mccImsAnalysisWrapper.preprocessing_options)
        preprocessing_parameters_dict = GCMSAnalysis.prepare_preprocessing_parameter_dict(mccImsAnalysisWrapper.preprocessing_options)
        preprocessing_steps_dict['all_peak_detection_steps'] = preprocessing_steps_dict['peak_detection_steps']

    else:
        preprocessing_steps_dict = MccImsAnalysis.match_processing_options(mccImsAnalysisWrapper.preprocessing_options)
        preprocessing_parameters_dict = MccImsAnalysis.prepare_preprocessing_parameter_dict(mccImsAnalysisWrapper.preprocessing_options)
        preprocessing_steps_dict['all_peak_detection_steps'] = preprocessing_steps_dict['peak_detection_steps']
        preprocessing_steps_dict['all_peak_detection_steps'] += preprocessing_steps_dict['external_steps']

    # need to convert from enums to strings for passing to template
    context['dataset_info'] = dataset_info_dict
    # context['preprocessing_steps'] = preprocessing_steps_dict

    # need to sort by denoising etc
    denoising_parameter_dict = {k.name : v for k,v in preprocessing_parameters_dict.items() if k in preprocessing_steps_dict.get('denoising_steps', [])}
    normalization_parameter_dict = {k.name : v for k,v in preprocessing_parameters_dict.items() if k in preprocessing_steps_dict.get('normalization_steps',[])}

    all_peak_detection_parameter_dict = {}
    for k, v in preprocessing_parameters_dict.items():
        if k in preprocessing_steps_dict['all_peak_detection_steps']:
            all_peak_detection_parameter_dict[k.name] = v

            # sensitive info - location of zip
            if k.name == "VISUALNOWLAYER":
                if mccImsAnalysisWrapper.is_custom_analysis:
                    all_peak_detection_parameter_dict[k.name] = {}
                else:
                    path = Path(all_peak_detection_parameter_dict[k.name]['visualnow_filename'])
                    all_peak_detection_parameter_dict[k.name]['visualnow_filename'] = path.stem
            # sensitive info - peax binary
            elif k.name == "PEAX":
                all_peak_detection_parameter_dict[k.name] = {}

    peak_alignment_parameter_dict = {k.name : v for k,v in preprocessing_parameters_dict.items() if k == preprocessing_steps_dict['peak_alignment_step']}

    context['denoising_parameters'] = denoising_parameter_dict
    context['normalization_parameters'] = normalization_parameter_dict
    context['peak_detection_parameters'] = all_peak_detection_parameter_dict
    context['peak_alignment_parameters'] = peak_alignment_parameter_dict

    # evaluation_options
    evaluation_parameter_dict = {k : v for k,v in mccImsAnalysisWrapper.evaluation_options.items()}
    context['evaluation_parameters'] = evaluation_parameter_dict

    # prediction results
    web_prediction_model = WebPredictionModel.objects.all().filter(mcc_ims_analysis=mccImsAnalysisWrapper).first()
    if web_prediction_model:
        # get all results associated with web_prediciton model
        prediction_result_qs = PredictionResult.objects.all().filter(web_prediction_model=web_prediction_model.pk)

        result_dicts = []
        for pr in prediction_result_qs:
            # import pdb; pdb.set_trace()
            orig_labels = {}
            if pr.original_class_labels:
                orig_labels = pr.original_class_labels
            result_dicts.append({
                # add orig_label default "" to match new table style
                "class_assignment": [(m_name, label, orig_labels.get(m_name, "")) for m_name, label in
                                     pr.class_assignment.items()],
                "peak_detection_method_name": pr.peak_detection_method_name,
            })
        context['prediction_parameters'] = result_dicts

    return context

def create_preprocessing_params_from_session_forms(dataset_info):
    plot_params, file_params = construct_default_parameters(file_prefix=dataset_info['dataset_name'],
                                                            folder_name=dataset_info['dataset_name'])
    return {"plot_params": plot_params,
            "file_params ": file_params}

def create_evaluation_params_from_form(analysis_form):
    # map parameter names to values that match the naming in BreathCore.MccImsAnalysis.DEFAULT_EVALUATION_MEASURE_PARAMETERS
    # create dicts here from name of evaluaion_measures to params
    analysis_matcher = AnalysisFormMatcher(analysis_form)
    return analysis_matcher.performance_measure_parameters



# @login_required
@temp_or_login_required
@csrf_protect
def review(request, user):
    template_name = 'breath/review.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)
    if request.session['dataset_info'] and request.session['run_parameters']:
        # session parameters are available in templates due to context processor settings
        # give user the ability to go back to other steps
        if request.method == "POST":
            if 'dataset' in request.POST:
                # go back to dataset selection - clear session variables
                request.session.pop('dataset_info')
                request.session.pop('run_parameters')
                request.session.pop('preprocessing_parameters')
                return redirect(selectDataset)

            elif 'processing_steps' in request.POST:
                # go back to preprocessing selection - clear session variables
                request.session.pop('run_parameters')
                return redirect(selectParameters)

            else: # start the analysis
                request.session['start_preprocessing'] = True
                ims_set = get_object_or_404(WebImsSet, pk=request.session['dataset_info']['ims_set_pk'])

                # if visualnow selected, then edit preprocessing param so that it holds the absolute path, not only the filename
                if PeakDetectionMethod.VISUALNOWLAYER.name in request.session['preprocessing_parameters'] and request.session['dataset_info'].get('peak_layer_filename', ''):
                    absolute_path_visualnow_layer = f"{ims_set.upload.path}/{request.session['dataset_info']['peak_layer_filename']}"
                    request.session['preprocessing_parameters'][PeakDetectionMethod.VISUALNOWLAYER.name] = {'visualnow_filename': absolute_path_visualnow_layer}

                analysis_wrapper = MccImsAnalysisWrapper(ims_set=ims_set,
                                                  preprocessing_options=request.session['preprocessing_parameters'],
                                                  evaluation_options={},
                                                  class_label_mapping={},
                                                  user=user)
                analysis_wrapper.save()
                # print(ims_model.preprocessing_options)
                analysis_id = analysis_wrapper.pk
                # request.session['analysis_id'] = analysis_id
                return redirect(f'analysis/{analysis_id}')

        else: # GET request - we render the datatable and the form
            review_form = CrispyReviewForm()
            context['review_form'] = review_form

            # create temporary analysis object fo use of function
            ims_set = get_object_or_404(WebImsSet, pk=request.session['dataset_info']['ims_set_pk'])
            analysis_wrapper = MccImsAnalysisWrapper(
                    ims_set=ims_set, preprocessing_options=request.session['preprocessing_parameters'],
                    evaluation_options={}, class_label_mapping={}, user=user)
            analysis_wrapper.save()
            context.update(prepare_analysis_details_for_context(analysis_wrapper, add_images=False))
            analysis_wrapper.delete()
            return render(request, template_name, context=context)

    # in case something went wrong, or session expired - redirect to start of process
    return redirect('selectDataset')


@csrf_protect
def _selectDataset(request, user, template_name='breath/selectdataset.html', on_success_redirect_to='selectParameters'):
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)

    if request.method == "POST":
        web_ims_set_form = WebImsSetForm(request.POST, user_id=user.pk)
    # if GET or other request create default form
    else:
        web_ims_set_form = WebImsSetForm(user_id=user.pk)
        context['form'] = web_ims_set_form
        return render(request, template_name, context)

    if web_ims_set_form.is_valid():
        # save info from dataset in session - bad practice - users can only create one job at a time
        empty_fileset = FileSet(name=web_ims_set_form.cleaned_data['zip_file'].name[:50])
        empty_fileset.save()

        # reuse defined Fileset in analysis, and dont recreate one
        web_ims_set_model = WebImsSet(upload=web_ims_set_form.cleaned_data['zip_file'], file_set=empty_fileset, user=user)
        web_ims_set_model.save()

        dataset_info = web_ims_set_model.get_dataset_info()

        # actually create a model object and save it in the ORM
        dataset_info['ims_set_pk'] = web_ims_set_model.pk
        print(f"Created {dataset_info} ")
        request.session['dataset_info'] = dataset_info
        if 'not_valid' in context:
            context.pop('not_valid')
        return redirect(on_success_redirect_to)
    else:
        context['not_valid'] = "Yes"
        context['form'] = web_ims_set_form
    return render(request, template_name, context)


@temp_or_login_required
@csrf_exempt
def selectDataset(request, user):
    # required due to handling of files in different way when they have different sizes, we always want a temporary file for zips, not an in memory file
    # see https://stackoverflow.com/questions/6906935/problem-accessing-user-uploaded-video-in-temporary-memory
    # and https://docs.djangoproject.com/en/dev/topics/http/file-uploads/#modifying-upload-handlers-on-the-fly
    # removes the first file handler (MemoryFile....)
    # pop all session stuff
    pop_non_user_session_keys(request)
    return _selectDataset(request, user=user)

# @login_required
@temp_or_login_required
def selectParameters(request, user):
    template_name = 'breath/selectparameters.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)

    if request.session['dataset_info']:
        # only process POST requests
        if request.method == 'POST':
            processing_form = ProcessingStepsForm(request.POST)
            # validate Form
            if processing_form.is_valid():
                peak_layer_filename = request.session['dataset_info'].get('peak_layer_filename', '')
                processingStepsFormMatcher = ProcessingStepsFormMatcher(processing_form,
                                                                        peak_layer_filename=peak_layer_filename,
                                                                        peax_binary_path=settings.PEAX_BINARY_PATH
                                                                        )
                preprocessing_parameters = processingStepsFormMatcher.preprocessing_parameters
                # used to visualize in review
                run_parameters = dict()
                run_parameters['peak_detection'] = [pdm.name for pdm in processing_form.cleaned_data['peak_detection']]
                run_parameters['peak_alignment'] = [processing_form.cleaned_data['peak_alignment'].name]
                run_parameters['denoising_method'] = [dnm.name for dnm in processing_form.cleaned_data['denoising_method']]

                # log something in session, so we know all previous steps were completed
                request.session['run_parameters'] = run_parameters
                request.session['preprocessing_parameters'] = preprocessing_parameters
                return redirect('review')
            else:
                context['not_valid'] = "Yes"
                context['processing_form'] = processing_form
                return render(request, template_name, context)

        # if get or other request create default form
        else:
            # Initial values are set in class definition
            processing_form = ProcessingStepsForm()
            context['processing_form'] = processing_form
            return render(request, template_name, context)
    else:
        # TODO give error message to tell user what went wrong
        return redirect('selectdataset')


# @login_required
@temp_or_login_required
def automatic_analysis_progress(request, task_id, analysis_id, user):

    # here we handle the further automatic processing - when result is ready we redirect to prediction/<analysis_id>
    # analysis_id = request.session['analysis_id']
    # check if user has access to check the status, or if we shared the task_id
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)


    prediction_url = reverse('prediction', kwargs={'analysis_id' : analysis_id})
    # "celery-tasks"
    task_url = reverse('task_status', kwargs={'task_id': task_id})
    context = {'task_id': task_id, 'result_url': prediction_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, template_name='breath/display_progress.html', context=context)


@temp_or_login_required
def analysis_progress(request, task_id, analysis_id, user):
    # from .tasks import full_parallel_preprocessing
    # result = full_parallel_preprocessing.delay(analysis_id)
    # analysis_id = request.session['analysis_id']
    # check if user has access to check the status, or if we shared the task_id
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    analysis_url = reverse('analysis', kwargs={'analysis_id' : analysis_id})
    print(analysis_url)
    # "celery-tasks"
    task_url = reverse('task_status', kwargs={'task_id': task_id})
    context = {'task_id': task_id, 'result_url': analysis_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, template_name='breath/display_progress.html', context=context)


@temp_or_login_required
def evaluate_performance_progress(request, task_id, analysis_id, user):
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    prediction_url = reverse('prediction', kwargs={'analysis_id' : analysis_id})
    print(prediction_url)
    task_url = reverse('task_status', kwargs={'task_id': task_id})
    context = {'task_id': task_id, 'result_url': prediction_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, 'breath/display_progress.html', context=context)


@temp_or_login_required
def prediction_progress(request, task_id, analysis_id, user):
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    prediction_result_url = reverse('prediction_result', kwargs={'analysis_id': analysis_id})
    # print(prediction_result_url)
    task_url = reverse('task_status', kwargs={'task_id': task_id})

    context = {'task_id': task_id, 'result_url': prediction_result_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, 'breath/display_progress.html', context=context)


@temp_or_login_required
def custom_prediction_progress(request, task_id, analysis_id, user):
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)
    prediction_result_url = reverse('custom_prediction_result', kwargs={'analysis_id': analysis_id})

    task_url = reverse('task_status', kwargs={'task_id': task_id})

    context = {'task_id': task_id, 'result_url': prediction_result_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, 'breath/display_progress.html', context=context)


@temp_or_login_required
def custom_evaluation_progress(request, task_id, analysis_id, user):
    # analysis_id = request.session['analysis_id']
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    prediction_url = reverse('custom_prediction', kwargs={'analysis_id': analysis_id})
    # print(prediction_result_url)
    task_url = reverse('task_status', kwargs={'task_id': task_id})

    context = {'task_id': task_id, 'result_url': prediction_url, 'task_url': task_url}
    context = if_temp_update_context(user=user, context=context)
    return render(request, 'breath/display_progress.html', context=context)


@temp_or_login_required
def analysis(request, analysis_id, user):
    """
    Preprocess measurements if redirect from review and session.start_preprocessing
    if not session then render form
    if POST start evaluation task
    :param request:
    :param analysis_id:
    :param user:
    :return:
    """
    from .tasks import EvaluatePerformanceTask, ParallelPreprocessingTask
    # from django.http import Http404,
    # from celery.v
    template_name = 'breath/analysis.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)

    # # need to check if user_id associated with analysis
    # Make sure the analysis id actually exists
    mcc_ims_wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)
    # assert isinstance(mcc_ims_wrapper, MccImsAnalysisWrapper)  # asserts will only be called if debug is true

    # start preprocessing
    if request.session.get('start_preprocessing', '') and request.session['dataset_info']:# and request.session['run_parameters']:
        # now start pipeline

        # If we can't connect to the backend, let's not just return a 500 error code
        # result = SampleTask.delay_or_fail(seconds=5)
        request.session.pop('start_preprocessing')
        # new parallel implementation
        # result = PreprocessingTask.delay_or_fail(analysis_id)
        result = ParallelPreprocessingTask.delay_or_fail(analysis_id=analysis_id)

        return redirect(
            reverse('analysis_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

    # if request.session.get('finished_preprocessing', ''):
    minimum_occurences_class_label = 0
    sorted_occurences = sorted(Counter(mcc_ims_wrapper.class_label_mapping.values()).values())
    if sorted_occurences:
        minimum_occurences_class_label = sorted_occurences[0]

    if request.method == "POST":
        # request.session['start_evaluation'] = True
        analysis_form = AnalysisForm(minimum_occurences_class_label, request.POST)

        # validate Form
        if analysis_form.is_valid():

            # get all parameters from analysis form and update evaluation parameters in WebImsSet
            mcc_ims_wrapper.evaluation_options = create_evaluation_params_from_form(analysis_form)
            mcc_ims_wrapper.save()

            # start async  EvaluatePerformanceTask and redirect to polling point
            # result = EvaluatePerformanceTask().delay_or_fail(analysis_id=analysis_id)
            result = EvaluatePerformanceTask().delay_or_fail(analysis_id=analysis_id)

            return redirect(
                reverse('evaluate_performance_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

        else:
            context['not_valid'] = "Yes"
            context['analysis_form'] = analysis_form
    else:
        analysis_form = AnalysisForm(minimum_occurences_class_label=minimum_occurences_class_label)
        analysis_form.helper.form_action = reverse('analysis', kwargs={"analysis_id": analysis_id})
        context['analysis_form'] = analysis_form

        plot_retriever = PlotRetriever(analysis_id)
        # plain_images = plot_retriever.get_plots_of_analysis([ClusterPlotModel, IntensityPlotModel, OverlayPlotModel])

        context['images'], context['available_tags'] = plot_retriever.get_plots_of_analysis(
            [ClusterPlotModel, ClasswiseHeatMapPlotModel, OverlayPlotModel])

    return render(request, template_name, context=context)


@temp_or_login_required
def predictor_pickle_download(request, analysis_id, user):
    # get location? of TrainingsMatrix, statisticsDF, PredictionResult and PredictionModel matching analysis_id

    # only allow correct user to access
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)
    prediction_result = WebPredictionModel.objects.filter(mcc_ims_analysis=analysis_id)[0]
    pickled_predictor = prediction_result.scipy_predictor_pickle
    filename = prediction_result.scipy_predictor_pickle.file.name.split('/')[-1] + ".pickle"

    response = HttpResponse(pickled_predictor.file, content_type='application/octet-stream')
    response['Content-Disposition'] = 'attachment; filename=%s' % filename
    return response


def about(request):
    template_name = 'breath/about.html'
    context = {'active_page': 'about',
               }
    context = update_user_context_from_request(request, context)
    return render(request, template_name, context=context)

def documentation(request):
    template_name = 'breath/documentation.html'
    context = {'active_page': 'documentation',
               }
    context = update_user_context_from_request(request, context)
    return render(request, template_name, context=context)

def help(request):
    template_name = 'breath/help.html'
    context = {'active_page': 'help',
               }
    context = update_user_context_from_request(request, context)
    return render(request, template_name, context=context)


@temp_or_login_required
@csrf_exempt
def prediction(request, analysis_id, user):
    # required due to handling of files in different way when they have different sizes, we always want a temporary file for zips, not an in memory file
    # see https://stackoverflow.com/questions/6906935/problem-accessing-user-uploaded-video-in-temporary-memory
    # and https://docs.djangoproject.com/en/dev/topics/http/file-uploads/#modifying-upload-handlers-on-the-fly
    # removes the first file handler (MemoryFile....)
    # request.upload_handlers.pop(0)
    return _prediction(request, analysis_id, user)


@csrf_protect
def _prediction(request, analysis_id, user):
    from django.core.exceptions import MultipleObjectsReturned
    from .tasks import PredictClassTask
    template_name = 'breath/prediction.html'
    context = {'active_page': 'prediction',
               }
    print("Reached prediction - ")
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
        print(f"Got mulitple WebPredictionresults, defaulting to {prediction_model}")

    # prediction_model_pk = WebPredictionModel.objects.filter()[0].pk
    # prediction_model_pk = WevaebPredictionModel.objects.filter()[0].pk
    request.session['prediction_model_pk'] = prediction_model.pk

    # get_object_or_404(WebPredictionModel, pk=prediction_model_id)
    # if set, we can continue
    prediction_model_pk = prediction_model.pk
    # removed analysis id from session to allow multiple runs at once per user
    # request.session['analysis_id'] = analysis_id
    if request.method == "POST":
        # form = UploadZipfileForm(request.POST, request.FILES)
        if request.FILES:
            prediction_form = PredictionForm(prediction_model_pk, request.POST, request.FILES, user_id=user.pk)
        else:
            prediction_form = PredictionForm(prediction_model_pk, request.POST, user_id=user.pk)
        if prediction_form.is_valid():
            classPredictionFileSet_id = prediction_form.cleaned_data['prediction_file_set_id']
            result = PredictClassTask().delay_or_fail(analysis_id=analysis_id, prediction_file_set_id=classPredictionFileSet_id)

            return redirect(
                reverse('prediction_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

        else:
            context['form_errors'] = prediction_form.errors

    context = prepare_prediction_template_parameters(context, analysis_id, user, mcc_ims_analysis_wrapper.is_custom_analysis)

    prediction_form = PredictionForm(prediction_model_pk, user_id=user.pk)
    context['prediction_form'] = prediction_form
    return render(request, template_name, context=context)


@temp_or_login_required
def prediction_result(request, analysis_id, user):

    template_name = 'breath/prediction_result.html'
    context = {'active_page': 'prediction',
               }

    # analysis_id = request.session.get('analysis_id', '')
    # get prediction model matching analysis from DB
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    web_prediction_model = get_object_or_404(WebPredictionModel, mcc_ims_analysis=wrapper)
    # pdb.set_trace()

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
    if wrapper.is_automatic_analysis:
        images, available_tags = pr.get_plots_of_analysis(
            [BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel, DecisionTreePlotModel], limit_to_peak_detection_method_name=wrapper.automatic_selected_method_name)
    else:
        images, available_tags = pr.get_plots_of_analysis(
            [BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel, DecisionTreePlotModel])
    context['images'] = images
    context['available_tags'] = available_tags
    context['analysis_id'] = analysis_id
    return render(request, template_name, context=context)


@temp_or_login_required
def custom_prediction_result(request, analysis_id, user):

    template_name = 'breath/custom_prediction_result.html'
    context = {'active_page': 'prediction',
               }

    # analysis_id = request.session.get('analysis_id', '')
    # get prediction model matching analysis from DB
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    web_prediction_model = get_object_or_404(WebPredictionModel, mcc_ims_analysis=wrapper)
    # pdb.set_trace()

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
    if wrapper.is_automatic_analysis:
        images, available_tags = pr.get_plots_of_analysis(
            [BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel, DecisionTreePlotModel], limit_to_peak_detection_method_name=wrapper.automatic_selected_method_name)
    else:
        images, available_tags = pr.get_plots_of_analysis(
            [BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel, DecisionTreePlotModel])
    context['images'] = images
    context['available_tags'] = available_tags
    context['analysis_id'] = analysis_id
    return render(request, template_name, context=context)


class PlotRetriever(object):
    """
    retrieve plots by their analysis and plot type
    sort them into dict according to their PlotType
    resulting is a list with Image
    """

    def __init__(self, analysis_id):
        self.analysis_id = analysis_id

    def get_plots_of_analysis(self, model_instance_list, limit_to_peak_detection_method_name=False):
        plots = []

        for model_instance in model_instance_list:
            try:
                model_has_field = model_instance._meta.get_field('based_on_peak_detection_method_name')
            except FieldDoesNotExist:
                model_has_field = False
                pass
            if limit_to_peak_detection_method_name and model_has_field:
                plot_qs = model_instance.objects.all().filter(analysis=self.analysis_id, based_on_peak_detection_method_name=limit_to_peak_detection_method_name)
            else:
                plot_qs = model_instance.objects.all().filter(analysis=self.analysis_id)

            plots.extend([self.create_plot_holder(plot_instance=plot_instance) for plot_instance in plot_qs if plot_instance.__class__ == model_instance])
            # just add plots to list, are already tagged


        # make tags for plots for filtering
        # new tagging using better js filtering
        all_tags = []
        for img in plots:
            all_tags.extend(img['tags'])
        all_available_tags = np.unique(all_tags).tolist()
        # remove all PeakIds - potentially use as secondary tags
        primary_tags = [tag for tag in all_available_tags if not tag.startswith("Peak_")]

        # sort alphabetically - capitalization shouldnt matter
        sorted_primary_tags = np.array(primary_tags)[np.argsort([str.lower(tag) for tag in primary_tags])].tolist()

        return plots, sorted_primary_tags

    @staticmethod
    def _match_image_tags(class_instance):
        instance_tag_matcher = dict()
        instance_tag_matcher[RocPlotModel] = ['ROC', ]
        instance_tag_matcher[ClusterPlotModel] = ['cluster', ]
        instance_tag_matcher[BoxPlotModel] = ['box_plot', ]
        instance_tag_matcher[DecisionTreePlotModel] = ['decision_tree', ]
        instance_tag_matcher[IntensityPlotModel] = ['intensity', ]
        instance_tag_matcher[ClasswiseHeatMapPlotModel] = ['intensity', ]
        instance_tag_matcher[OverlayPlotModel] = ['overlay', ]
        instance_tag_matcher[BestFeaturesOverlayPlot] = ['overlay', 'best_features']
        return instance_tag_matcher[class_instance]

    @staticmethod
    def _match_image_title(plot_model_instance):

        def format_peak_detection_method_name(pdmn):
            if pdmn == PeakDetectionMethod.VISUALNOWLAYER.name:
                pdmn_abrrev = "by_layer"
            else:
                pdmn_abrrev = str.capitalize(str.lower(pdmn))
            return pdmn_abrrev

        pdmn_abrev = format_peak_detection_method_name(plot_model_instance.based_on_peak_detection_method_name)

        if isinstance(plot_model_instance, BestFeaturesOverlayPlot):
            image_title = f"Best Peaks {pdmn_abrev}"
            plot_description = f"This plot shows the best n selected peaks using the {pdmn_abrev} peak detection method."

        elif isinstance(plot_model_instance, OverlayPlotModel):
            image_title = f"Peaks detected by {pdmn_abrev}"
            plot_description = f"This plot shows the peaks detected by {pdmn_abrev} as peak detection method."

        elif isinstance(plot_model_instance, RocPlotModel):
            image_title = f"ROC AUC {pdmn_abrev}"
            plot_description = f"This plot shows the ROC curves for each of the n-fold cross validation runs. This value can be used to estimate model performance."

        elif isinstance(plot_model_instance, BoxPlotModel):
            image_title = f"Boxplot {pdmn_abrev} {plot_model_instance.based_on_peak_id}"
            plot_description = f"This plot shows the boxplots of normalized peak intensities for each respective class. The corrected p_values are calculated with the Mann-Whitney-U test and corrected using Benjamini-Hochberg FDR cutoff."

        elif isinstance(plot_model_instance, ClusterPlotModel):
            image_title = f"Peak Clusters {pdmn_abrev}"
            plot_description = f"This plot shows the peak positions for peaks detected by {pdmn_abrev} and clustered using the selected peak-alignment strategy."

        elif isinstance(plot_model_instance, ClasswiseHeatMapPlotModel):
            image_title = f"Averaged Intensity by {plot_model_instance.class_label}"
            plot_description = f"This plot shows the average intensities of all measurements of class {plot_model_instance.class_label}."

        elif isinstance(plot_model_instance, DecisionTreePlotModel):
            pmn = str(plot_model_instance.based_on_performance_measure_name)
            image_title = f"Decision-Tree {pdmn_abrev} {pmn}"
            plot_description = f"This plot shows the Decision-Tree guiding the classification process. Nodes are colored by a mixture of the colors assigned with each class and visualize the class representation in the nodes below this split. Use right-click 'View Image' if the text is too small."
        else:
            # basic image title
            image_title = f"{str(plot_model_instance.__class__)} {pdmn_abrev}"
            plot_description = f"This plot doesn't seem to have a description"


        return image_title, plot_description


    def create_plot_holder(self, plot_instance):
        p = plot_instance
        image_tags = []
        rv = {'name': p.name, 'url': p.figure.url,
              # be careful to use app urls when serving media - media_url is project wise and doesnt show up
            'src_tag': f"src={reverse('index')}{p.figure.url}"}

        image_tags.extend(self._match_image_tags(plot_instance.__class__))

        image_title, plot_description = self._match_image_title(plot_instance)

        if isinstance(plot_instance, HeatmapPlotModel):
            measurement_name = str(plot_instance.based_on_measurement)
            rv['measurement_name'] = measurement_name
            if measurement_name:
                image_tags.append(measurement_name)

            # check if we have a best_by_class plot
            if "best_by_class-" in measurement_name:
                class_label = measurement_name.split("best_by_class-")[-1]
                image_tags.append(class_label)
                image_tags.append("best_by_class")
                image_title += f" {class_label}"

        # add tag by pdmn - Intensity plots are not associated, so exempt
        if not (plot_instance.__class__ == IntensityPlotModel or plot_instance.__class__ == ClasswiseHeatMapPlotModel):
            image_tags.extend([plot_instance.based_on_peak_detection_method_name])

        if plot_instance.__class__ == ClasswiseHeatMapPlotModel:
            image_tags.append(str(plot_instance.class_label))
            image_tags.append("best_by_class")

        if plot_instance.__class__ == ClusterPlotModel:
            image_tags.append(str(plot_instance.based_on_peak_alignment_method_name))

        if plot_instance.__class__ in [BoxPlotModel, RocPlotModel, BestFeaturesOverlayPlot, DecisionTreePlotModel]:
            image_tags.append(str(plot_instance.based_on_performance_measure_name))

        if plot_instance.__class__ in [BoxPlotModel,]:
            image_tags.append(str(plot_instance.based_on_peak_id))

        rv['tags'] = image_tags
        rv['image_title'] = image_title
        rv['plot_description'] = plot_description

        return rv

    def get_zip_of_plot_models(self, model_instance_list):
        """
        Zip plots for analysis into single archive
        :param model_instance_list:
        :return: zip-archive
        """
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, ZIP_DEFLATED, "w") as zip_out:
            for model_instance in model_instance_list:
                plot_qs = model_instance.objects.all().filter(analysis=self.analysis_id)
                for plot_model in plot_qs:
                    # get png from plot
                    plot_full_fn = plot_model.figure.path
                    plot_archive_fn = Path(plot_full_fn).name
                    # somehow we need to call the buffer instead of the file handle, otherwise utf8 encoding error
                    with open(plot_full_fn, "r") as p_open:
                        zip_out.writestr(plot_archive_fn, p_open.buffer.read())
        return zip_buffer

class HttpResponseUnauthorized(HttpResponse):
    status_code = 401


@temp_or_login_required
@csrf_exempt
def custom_detection_analysis(request, user):
    # required due to handling of files in different way when they have different sizes, we always want a temporary file for zips, not an in memory file
    # see https://stackoverflow.com/questions/6906935/problem-accessing-user-uploaded-video-in-temporary-memory
    # and https://docs.djangoproject.com/en/dev/topics/http/file-uploads/#modifying-upload-handlers-on-the-fly
    return _custom_detection_analysis(request, user=user)

@csrf_protect
def _custom_detection_analysis(request, user):
    """
    Enable the user to upload custom peak-detection results for proper cross validation and plots, makes possible to support other MS methods
    Should upload zip file with peak detection results and class label dict
    :param request:
    :param user:
    :return:
    """
    from .models import CustomPeakDetectionFileSet
    template_name = 'breath/custom_detection_analysis.html'
    context = {'active_page': 'prediction',
               }
    print("Reached custom detection analysis")

    context = if_temp_update_context(user=user, context=context)

    if request.method == "POST":
        # form = UploadZipfileForm(request.POST, request.FILES)
        custom_peak_detection_analysis_form = CustomDetectionAnalysisForm(request.POST, user_id=user.pk)
        if custom_peak_detection_analysis_form.is_valid():

            using_feature_matrix = custom_peak_detection_analysis_form.cleaned_data.get('feature_matrix_id', False)

            if using_feature_matrix:
                feature_matrix_id = using_feature_matrix
                custom_feature_matrix = get_object_or_404(FeatureMatrix, pk=feature_matrix_id)
                samples_per_class = Counter(custom_feature_matrix.class_label_dict.values())
                minimum_observations = min(samples_per_class.values())

                coerced_pdm = custom_feature_matrix.get_used_peak_detection_method()

            else:
                # CustomPeakDetectionFileSet
                customPDFS_id = custom_peak_detection_analysis_form.cleaned_data['customPDFS_id']
                # create Task to process custom peak detection
                # create Analysis to associate with - all plots need Analysis id and all other things are related
                # have separate file management - but use all other data structures
                # Auto select evaluation options - no need for manual interaction - 5 fold cv as standard if possible from sample size
                # then get metrics for evaluation and possibility to do upload more custom pdr for application of model
                customPDFS = get_object_or_404(CustomPeakDetectionFileSet, pk=customPDFS_id)

                # create WebCustomSet from custom Fileset
                custom_web_set = WebCustomSet(file_set=customPDFS, user=user)
                custom_web_set.save()

                samples_per_class = Counter(customPDFS.class_label_processed_id_dict.values())
                minimum_observations = min(samples_per_class.values())

                coerced_pdm = customPDFS.get_used_peak_detection_method()


            # get default values
            preprocessing_options, eval_options = AnalysisFormMatcher.get_custom_preprocessing_evaluation_options(minimum_observations)

            # update with form values
            # use coerced peak detection method to update preprocessing options
            preprocessing_options.pop(ExternalPeakDetectionMethod.CUSTOM.name)
            preprocessing_options[coerced_pdm.name] = {}

            # form is valid - update preprocessing_options
            # make sure alignment is correct by popping old alignment
            preprocessing_options.pop(PeakAlignmentMethod.PROBE_CLUSTERING.name)

            preprocessing_options.update(custom_peak_detection_analysis_form.cleaned_data['preprocessing_parameters'])

            # distinction happened before submitting the eval params - this function
            if using_feature_matrix:
                custom_analysis = MccImsAnalysisWrapper(
                    ims_set=None, custom_set=None,
                    preprocessing_options=preprocessing_options, evaluation_options=eval_options,
                    class_label_mapping=custom_feature_matrix.class_label_dict, user=user,
                    is_custom_analysis=True, is_custom_feature_matrix_analysis=True, is_automatic_analysis=True,
                )
                custom_analysis.save()
                custom_feature_matrix.analysis = custom_analysis
                custom_feature_matrix.save()

                # result = CustomPeakDetectionFMAnalysisTask().delay_or_fail(analysis_id=custom_analysis.pk)

            # using peak detection results
            else:
                custom_analysis = MccImsAnalysisWrapper(
                    ims_set=None, custom_set=custom_web_set,
                    preprocessing_options=preprocessing_options, evaluation_options=eval_options,
                    class_label_mapping=customPDFS.class_label_processed_id_dict, user=user,
                    is_custom_analysis=True, is_automatic_analysis=True,
                )
                custom_analysis.save()
                # feature reduction was too stringent - reduced default percentage to 0.5
                # result = CustomPeakDetectionAnalysisTask().delay_or_fail(analysis_id=custom_analysis.pk)

            # return redirect(
            #     reverse('custom_evaluation_progress', kwargs={'task_id': result.task_id, 'analysis_id': custom_analysis.pk}))
            # redirect to evaluation parameter selection
            return redirect(
                reverse('custom_evaluation_params', kwargs={'analysis_id': custom_analysis.pk, "is_using_fm": int(using_feature_matrix)}))

            # custom_analysis_debug(custom_analysis.pk)

    else:
        custom_peak_detection_analysis_form = CustomDetectionAnalysisForm(user_id=user.pk)
    # custom_peak_detection_analysis_form.helper.form_action = reverse('analysis', kwargs={"analysis_id": analysis_id})
    context['custom_peak_detection_analysis_form'] = custom_peak_detection_analysis_form
    return render(request, template_name=template_name, context=context)

@temp_or_login_required
def custom_evaluation_params(request, analysis_id, is_using_fm, user):
    """
    Allow user to specify evaluation parameters for custom analysis with pdr and fm
    :param request:
    :param analysis_id:
    :param is_using_fm: int
    :param user:
    :return:
    """
    from .tasks import CustomPeakDetectionAnalysisTask, CustomPeakDetectionFMAnalysisTask

    template_name = 'breath/custom_select_evaluation.html'
    context = {'active_page': 'run',
               }
    context = if_temp_update_context(user=user, context=context)

    mcc_ims_wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)
    minimum_occurences_class_label = 0
    sorted_occurences = sorted(Counter(mcc_ims_wrapper.class_label_mapping.values()).values())
    if sorted_occurences:
        minimum_occurences_class_label = sorted_occurences[0]

    if request.method == "POST":
        # request.session['start_evaluation'] = True
        analysis_form = AnalysisForm(minimum_occurences_class_label, request.POST)

        # validate Form
        if analysis_form.is_valid():

            # get all parameters from analysis form and update evaluation parameters in WebImsSet
            mcc_ims_wrapper.evaluation_options = create_evaluation_params_from_form(analysis_form)
            mcc_ims_wrapper.save()

            # start async task and redirect to polling point

            if is_using_fm:
                result = CustomPeakDetectionFMAnalysisTask().delay_or_fail(analysis_id=analysis_id)
            else:
                result = CustomPeakDetectionAnalysisTask().delay_or_fail(analysis_id=analysis_id)

            return redirect(
                reverse('custom_evaluation_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

        else:
            context['not_valid'] = "Yes"
            context['analysis_form'] = analysis_form
    else:
        analysis_form = AnalysisForm(minimum_occurences_class_label=minimum_occurences_class_label)
        analysis_form.helper.form_action = reverse('custom_evaluation_params', kwargs={"analysis_id": analysis_id, "is_using_fm": is_using_fm})
        context['analysis_form'] = analysis_form

    return render(request, template_name=template_name, context=context)


def prepare_prediction_template_parameters(context, analysis_id, user, use_custom_prediction=False):
    """
    update context for prediction view
    :param analysis_id:
    :param user:
    :return:
    """
    mcc_ims_analysis_wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id, user=user)

    plot_retriever = PlotRetriever(analysis_id)

    if mcc_ims_analysis_wrapper.is_automatic_analysis:
        # less plots available for custom prediction / automatic prediction should only serve for best model
        if use_custom_prediction:
            images, available_tags = plot_retriever.get_plots_of_analysis(
                [RocPlotModel, BoxPlotModel, DecisionTreePlotModel],
                limit_to_peak_detection_method_name=mcc_ims_analysis_wrapper.automatic_selected_method_name)
        else:
            images, available_tags = plot_retriever.get_plots_of_analysis(
                [OverlayPlotModel, BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel, DecisionTreePlotModel],
                limit_to_peak_detection_method_name=mcc_ims_analysis_wrapper.automatic_selected_method_name)
    else:
        images, available_tags = plot_retriever.get_plots_of_analysis(
            [OverlayPlotModel, BestFeaturesOverlayPlot, RocPlotModel, BoxPlotModel, DecisionTreePlotModel])

    # TODO also add evaluation stats to analysis_details?
    context['stats_by_evaluation'] = mcc_ims_analysis_wrapper.prepare_evaluation_stats()


    if mcc_ims_analysis_wrapper.is_automatic_analysis:
        traings_matrix_qs = FeatureMatrix.objects.filter(analysis=analysis_id,
                                                         peak_detection_method_name=mcc_ims_analysis_wrapper.automatic_selected_method_name, is_training_matrix=True)
    else:
        traings_matrix_qs = FeatureMatrix.objects.filter(analysis=analysis_id, is_training_matrix=True)


    trainings_matrices = MccImsAnalysisWrapper.prepare_fm_for_template(feature_matrix_queryset=traings_matrix_qs)

    context['trainings_matrices'] = trainings_matrices
    context['reduced_trainings_matrices'] = MccImsAnalysisWrapper.prepare_reduced_fm_json_representation_list(analysis_id=analysis_id)
    # print(get_plots_of_analysis(model_instance=OverlayPlotModel, analysis_id=analysis_id))
    # add all matching images to context
    context['images'] = images
    context['available_tags'] = available_tags

    return context


@temp_or_login_required
def custom_prediction(request, analysis_id, user):
    from django.core.exceptions import MultipleObjectsReturned
    from .tasks import CustomPredictClassTask

    template_name = 'breath/custom_prediction.html'
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
        print(f"Got multiple WebPredictionResults, defaulting to {prediction_model}")

    # if set, we can continue
    prediction_model_pk = prediction_model.pk
    if request.method == "POST":
        custom_prediction_form = CustomPredictionForm(prediction_model_pk, request.POST, user_id=user.pk)

        if custom_prediction_form.is_valid():
            # classPredictionFileSet = ClassPredictionFileSet(upload=prediction_form.cleaned_data['zip_file'])
            # classPredictionFileSet.save()

            # differentiate when using custom_feature_matrix_analysis
            using_feature_matrix = custom_prediction_form.cleaned_data.get('feature_matrix_id', False)

            if using_feature_matrix:
                feature_matrix_id = using_feature_matrix
                # custom_feature_matrix = get_object_or_404(FeatureMatrix, pk=feature_matrix_id)
                result = CustomPredictClassTask().delay_or_fail(analysis_id=analysis_id, feature_matrix_id=feature_matrix_id)

            else:
                customPDFS_id = custom_prediction_form.cleaned_data['customPDFS_id']
                result = CustomPredictClassTask().delay_or_fail(analysis_id=analysis_id, customPDFS_id=customPDFS_id)

            return redirect(
                reverse('custom_prediction_progress', kwargs={'task_id': result.task_id, 'analysis_id': analysis_id}))

    # GET - we need to initialize the form
    else:
        custom_prediction_form = CustomPredictionForm(web_prediction_model_key=prediction_model_pk, user_id=user.pk)

    context = prepare_prediction_template_parameters(context=context, analysis_id=analysis_id, user=user, use_custom_prediction=True)
    context['custom_prediction_form'] = custom_prediction_form
    return render(request, template_name, context=context)


def pop_non_user_session_keys(request, exceptions=[]):
    """
    Pop non-authentication session keys to reset pipeline progress for custom analysis
    :param request:
    :return: `request.session`
    """
    keys_to_keep = ["user_id", "tmp_user_id", "not_first_visit"]
    if exceptions:
        keys_to_keep.extend(exceptions)
    current_keys = list(request.session.keys())
    for sk in current_keys:
        if sk not in keys_to_keep:
            request.session.pop(sk)

@temp_or_login_required
def get_trainings_matrix_as_csv(request, fm_id, user):
    # TODO get fm_id over post, nonetheless always checking for user permission
    #
    fm_object = FeatureMatrix.objects.get(pk=fm_id)
    # ensure user is authorized to view - trainingsMatrix model need user as field
    #   otherwise one can access others results by brute-forcing pks
    if not user == fm_object.get_owning_user():
        return HttpResponseForbidden()

    fm = fm_object.get_feature_matrix()

    buffer = StringIO()
    fm.to_csv(buffer, index=True, header=True, index_label="index")

    buffer.seek(0)
    response = HttpResponse(buffer, content_type='text/csv')

    if fm_object.is_training_matrix:
        train_test_prefix = "train"
    else:
        train_test_prefix = "test"
    response['Content-Disposition'] = f'attachment; filename={train_test_prefix}_{fm_object.name}.csv'

    return response


@temp_or_login_required
def get_trainings_matrix_as_json(request, fm_id, user):
    fm_object = FeatureMatrix.objects.get(pk=fm_id)
    # ensure user is authorized to view - trainingsMatrix model need user as field
    #   otherwise one can access others results by brute-forcing pks
    if not user == fm_object.get_owning_user():
        return HttpResponseForbidden()

    fm_json = fm_object.get_feature_matrix_data_json()

    return HttpResponse(fm_json, content_type="application/json")


@temp_or_login_required
def get_plot_archive(request, analysis_id, user):
    # make sure user is allowed to access the analysis
    wrapper = get_object_or_404(MccImsAnalysisWrapper, pk=analysis_id)
    if not user == wrapper.user:
        return HttpResponseForbidden()

    pr = PlotRetriever(analysis_id)
    plot_model_instances = [ClusterPlotModel, ClasswiseHeatMapPlotModel, BestFeaturesOverlayPlot, RocPlotModel,
                            BoxPlotModel, DecisionTreePlotModel]
    response = HttpResponse(pr.get_zip_of_plot_models(plot_model_instances).getvalue(), content_type="application/zip")
    response['Content-Disposition'] = f'attachment; filename=plots_analysis_{analysis_id}.zip'
    return response


#### from django-celery - django-celery not used for anything else
def task_status(request, task_id):
    """Returns task status and result in JSON format."""
    from celery.result import AsyncResult
    import celery.states as states
    from django.http import JsonResponse
    from celery.utils import get_full_cls_name
    from celery.utils.encoding import safe_repr

    result = AsyncResult(task_id)
    state, retval = result.state, result.result
    response_data = {'id': task_id, 'status': state, 'result': retval}
    if state in states.EXCEPTION_STATES:
        traceback = result.traceback
        response_data.update({'result': safe_repr(retval),
                              'exc': get_full_cls_name(retval.__class__),
                              'traceback': traceback})
    return JsonResponse({'task': response_data})

@temp_or_login_required
def delete_user_fileset(request, user, fs_id):
    """
    Delete a user owned fileset
    """
    # make sure user is allowed to access the fileset
    udfs = get_object_or_404(UserDefinedFileset, pk=fs_id, user=user)
    udfs.delete()
    return redirect('list_datasets')


@temp_or_login_required
def delete_user_feature_matrix(request, user, fm_id):
    """
    Delete a user owned feature matrix
    """
    # make sure user is allowed to access the fileset
    udfm = get_object_or_404(UserDefinedFeatureMatrix, pk=fm_id, user=user)
    udfm.delete()
    return redirect('list_datasets')


@temp_or_login_required
def download_user_feature_matrix_csv(request, fm_id, user):
    fm_object = UserDefinedFeatureMatrix.objects.get(pk=fm_id)
    # ensure user is authorized to view
    if not user == fm_object.user:
        return HttpResponseForbidden()

    fm = fm_object.get_feature_matrix()

    buffer = StringIO()
    fm.to_csv(buffer, index=True, header=True, index_label="index")

    buffer.seek(0)
    response = HttpResponse(buffer, content_type='text/csv')

    if fm_object.is_training_matrix:
        train_test_prefix = "train"
    else:
        train_test_prefix = "test"
    response['Content-Disposition'] = f'attachment; filename=User_feature_matrix_{train_test_prefix}_{fm_object.pk}.csv'

    return response

@temp_or_login_required
def download_user_fileset_zip(request, fs_id, user):
    # make sure user is allowed to access the analysis
    fs = get_object_or_404(UserDefinedFileset, pk=fs_id)
    if not user == fs.user:
        return HttpResponseForbidden()

    response = HttpResponse(fs.get_zip(), content_type="application/zip")
    if fs.is_train:
        train_val_str = "train"
    else:
        train_val_str = "validate"
    response['Content-Disposition'] = f'attachment; filename=user_fileset_{fs_id}_{fs.name.replace(" ","")}_{train_val_str}.zip'
    return response


@temp_or_login_required
def download_default_fileset_zip(request, fs_id, user):
    # make sure user is allowed to access the analysis
    fs = get_object_or_404(PredefinedFileset, pk=fs_id)
    # anyone can get the predefined sets

    response = HttpResponse(fs.upload.read(), content_type="application/zip")
    response['Content-Disposition'] = f'attachment; filename=default_fileset_{fs_id}.zip'
    return response

@temp_or_login_required
def download_pd_fileset_zip(request, fs_id, user):
    # make sure user is allowed to access the analysis
    fs = get_object_or_404(PredefinedCustomPeakDetectionFileSet, pk=fs_id)
    # anyone can get the predefined sets

    response = HttpResponse(fs.upload.read(), content_type="application/zip")
    response['Content-Disposition'] = f'attachment; filename=default_pd_fileset_{fs_id}.zip'
    return response

@temp_or_login_required
def download_fxml_fileset_zip(request, fs_id, user):
    # make sure user is allowed to access the analysis
    fs = get_object_or_404(GCMSPredefinedPeakDetectionFileSet, pk=fs_id)
    # anyone can get the predefined sets

    # response = HttpResponse(fs.get_zip.getvalue(), content_type="application/zip")
    response = HttpResponse(fs.upload.read(), content_type="application/zip")
    response['Content-Disposition'] = f'attachment; filename=default_fxml_fileset_{fs_id}.zip'
    return response
