import zipfile
from itertools import chain
import tempfile
import os
from pathlib import Path
from shutil import rmtree

from django import forms

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.files.storage import default_storage
from django.core.files.uploadedfile import InMemoryUploadedFile

from django.utils.translation import ugettext_lazy as _

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Field, Layout, Fieldset, ButtonHolder, Button, Div, HTML
from crispy_forms.bootstrap import AppendedText, PrependedAppendedText, PrependedText, TabHolder, Tab, FormActions

from .models import (WebPredictionModel, ZipFileValidator,
                     PredefinedFileset, PredefinedCustomPeakDetectionFileSet,
                     UserDefinedFileset, UserDefinedFeatureMatrix, AnalysisType,
                     GCMSFileSet, GCMSPeakDetectionFileSet, GCMSPeakDetectionResult,
                     GCMSPredefinedPeakDetectionFileSet, create_GCMS_peak_detection_fileset_from_zip, create_gcms_fileset_from_zip)

from breathpy.model.ProcessingMethods import (
    PeakDetectionMethod, ExternalPeakDetectionMethod, PeakAlignmentMethod, DenoisingMethod, NormalizationMethod,
    PerformanceMeasure, FeatureReductionMethod,
    GCMSPeakDetectionMethod, GCMSAlignmentMethod, GCMSPreprocessingMethod, GCMSSupportedDatatype
)

from breathpy.model.GCMSTools import (filter_mzml_or_mzxml_filenames, filter_feature_xmls)
from breathpy.model.BreathCore import (
    MccImsAnalysis, construct_custom_processing_evaluation_dict,
    GCMSAnalysis,
)
from breathpy.generate_sample_data import split_labels_ratio, write_raw_files_to_zip

REVIEW_OPTIONS = ['', 'DATASET', 'PROCESSING_STEPS']



class SignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    last_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2', )


def options_to_name(lis_options):
    return [opt.name for opt in lis_options]

class WebImsSetForm(forms.Form):
    """
    Needs to handle selection of predefined ims sets or upload of zipfile
    """
    # based on https://stackoverflow.com/questions/4789466/populate-a-django-form-with-data-from-database-in-view#answer-4789516
    based_on_predefined_sets = forms.ModelChoiceField(
        required=False,
        label="Predefined Datasets",
        queryset=PredefinedFileset.objects.all().order_by('name'),
    )

    user_file = forms.FileField(
        required=False,
        validators=[ZipFileValidator(max_size=2000 * 1024 * 1024, check_class_labels=True)],
        label="Own Dataset (Zip-File)",
        help_text="Please upload a zip archive containing the MCC-IMS-measurements for building the model and the class label file. The class label file has to be tab or comma separated.",
    )

    # def __init__(self, *args, user_id, **kwargs):
    def __init__(self, *args, **kwargs):
        super(WebImsSetForm, self).__init__(*args, **kwargs)

        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Submit'))

    def clean(self):
        # if both fields given, chose user submit
        based_on_predefined_sets = self.cleaned_data.get('based_on_predefined_sets')
        user_file = self.cleaned_data.get('user_file')
        zip_file = None

        if user_file:
            zip_file = user_file
            if isinstance(zip_file, InMemoryUploadedFile):

                filename = zip_file.name  # received file name
                file_obj = zip_file
                target_path = default_storage.path('tmp/')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)

                with default_storage.open('tmp/' + filename, 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)

                self.cleaned_data['zip_file_path'] = target_path + filename
            else:
                self.cleaned_data['zip_file_path'] = zip_file.temporary_file_path()
        elif based_on_predefined_sets and isinstance(based_on_predefined_sets, PredefinedFileset):
            # zip_file = self.construct_zip_file(based_on_predefined_sets)
            zip_file = based_on_predefined_sets.upload
            self.cleaned_data['zip_file_path'] = zip_file.path
        else:
            error_strings = ["Please select an available dataset or upload your own.",]
            for err in self.errors.values():
                # remove "* " representation from error
                error_strings.append(err.as_text()[2:])

            msg = ValidationError(error_strings)
            raise msg
            # self.add_error(based_on_predefined_sets, msg)
            # self.add_error(user_file, msg)
        self.cleaned_data['zip_file'] = zip_file

        return self.cleaned_data


    # make user_id available to model, this requires the view passing the user_id
    # see https://www.pydanny.com/adding-django-form-instance-attributes.html for reference
    # def __init__(self, *args, **kwargs):
    #     # self.user_id = user_id
    #     super(WebImsSetForm, self).__init__(*args, **kwargs)
    #     # set the user_id as attribute of the form
    #     # self._meta.model.user_id = user_id
    #
    #     # title = forms.CharField(max_length=50)
    #     #
    # def form_valid(self, form):
    #     # res = self.req
    #     # my_file = SimpleUploadedFile()
    #     # assert zipfile.is_zipfile()
    #     form.instance.user_id = self.request.user
    #     return super().form_valid()



class ProcessingStepsForm(forms.Form):
    PEAK_DETECTION_OPTIONS = options_to_name(chain(MccImsAnalysis.AVAILABLE_PEAK_DETECTION_METHODS,
                                   MccImsAnalysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS))
    PDO = []
    if ExternalPeakDetectionMethod.CUSTOM.name in PEAK_DETECTION_OPTIONS:
        PDO = [pdo for pdo in PEAK_DETECTION_OPTIONS if pdo != ExternalPeakDetectionMethod.CUSTOM.name]
    PEAK_DETECTION_OPTIONS = PDO

    DENOISING_METHODS = options_to_name(chain(MccImsAnalysis.AVAILABLE_NORMALIZATION_METHODS,
        MccImsAnalysis.AVAILABLE_DENOISING_METHODS,
                                              ))
    # ['WATERSHED', 'TOPHAT', 'JIBB', 'PEAX', 'VISUALNOWLAYER']
    @staticmethod
    def _coerce_peak_detection(str):
        try:
            rv = PeakDetectionMethod(str)
        except ValueError:
            rv = ExternalPeakDetectionMethod(str)
        return rv

    @staticmethod
    def _coerce_denoising_method(str):
        try:
            rv = DenoisingMethod(str)
        except ValueError:
            rv = NormalizationMethod(str)
        return rv

    # BASIC OPTIONS
    peak_detection = forms.TypedMultipleChoiceField(
        help_text="Select one or multiple peak detection method(s).",
        label="Peak Detection",
        choices=zip(PEAK_DETECTION_OPTIONS, PEAK_DETECTION_OPTIONS),
        # coerce=lambda x: PeakDetectionMethod(x) or ExternalPeakDetectionMethod(x),
        coerce=lambda x: ProcessingStepsForm._coerce_peak_detection(x),
        widget=forms.CheckboxSelectMultiple,
        initial=[PeakDetectionMethod.TOPHAT.name, PeakDetectionMethod.VISUALNOWLAYER.name, PeakDetectionMethod.WATERSHED, ExternalPeakDetectionMethod.PEAX.name],
        # initial=[PeakDetectionMethod.TOPHAT.name, PeakDetectionMethod.WATERSHED, ExternalPeakDetectionMethod.PEAX.name],
        required=True,
        )

    peak_alignment = forms.TypedChoiceField(
        help_text="Select peak alignment method.",
        label="Peak Alignment",
        choices=zip(options_to_name(MccImsAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS),
            options_to_name(MccImsAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS)),
        coerce=lambda x: PeakAlignmentMethod(x),
        # widget=forms.CheckboxInput,
        initial=PeakAlignmentMethod.PROBE_CLUSTERING.name,
        required=True,
    )

    denoising_method = forms.TypedMultipleChoiceField(
        help_text="Select normalization / denoising method(s).",
        label="Normalization / Denoising",
        choices=zip(DENOISING_METHODS, DENOISING_METHODS),
        coerce=lambda x: ProcessingStepsForm._coerce_denoising_method(x),
        widget=forms.CheckboxSelectMultiple,
        initial=options_to_name([DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY, DenoisingMethod.MEDIAN_FILTER, DenoisingMethod.GAUSSIAN_FILTER, DenoisingMethod.SAVITZKY_GOLAY_FILTER, DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION, NormalizationMethod.INTENSITY_NORMALIZATION, NormalizationMethod.BASELINE_CORRECTION]),
    )

    # ADVANCED OPTIONS
    # Peak detection advanced parameters
    tophat_noise_threshold = forms.FloatField(
        label="TOPHAT Noise Threshold",
        required=False,
        min_value=1.0,
        max_value=5.0,
        initial=1.4,
        help_text="Select the intensity factor below which everything is considered noise.",
    )

    watershed_noise_threshold = forms.FloatField(
        label="WATERSHED Noise Threshold",
        required=False,
        min_value=1.0,
        max_value=5.0,
        initial=1.5,
        help_text="Select the intensity factor below which everything is considered noise.",
    )

    jibb_noise_threshold = forms.FloatField(
        label="JIBB Noise Threshold",
        required=False,
        min_value=1.0,
        max_value=5.0,
        initial=1.5,
        help_text="Select the intensity factor below which everything is considered noise.",
    )

    jibb_range_ivr = forms.IntegerField(
        label="JIBB #spectra width",
        required=False,
        min_value=3,
        max_value=21,
        initial=5,
        help_text="Number of values that need to increase steadily on the x-axis.",
    )


    jibb_range_rt = forms.IntegerField(
        label="JIBB #spectra height",
        required=False,
        min_value=3,
        max_value=21,
        initial=7,
        help_text="Number of values that need to increase steadily on the y-axis.",
    )

    # Peak Alignment advanced params
    dbscan_eps = forms.FloatField(
        label="DBSCAN eps",
        required=False,
        min_value=0.01,
        max_value=0.5,
        initial=0.025,
        help_text="Distance from the core sample, above this considered outlier.",
    )


    dbscan_no_sample = forms.IntegerField(
        label="DBSCAN #Min Samples",
        required=False,
        min_value=1,
        max_value=50,
        initial=2,
        help_text="Minimum number of samples required for a cluster.",
    )

    probe_clustering_stepwidth_irm = forms.FloatField(
        label="PROBE_CLUSTERING Stepwidth X",
        required=False,
        min_value=0.001,
        max_value=0.1,
        initial=MccImsAnalysis.BASIC_THRESHOLD_INVERSE_REDUCED_MOBILITY,
        help_text="Gridwidth for inverse reduced ion mobility.",
    )

    probe_clustering_scaling_rt = forms.FloatField(
        label="PROBE_CLUSTERING Scaling Y",
        required=False,
        min_value=0.01,
        max_value=0.5,
        initial=MccImsAnalysis.BASIC_THRESHOLD_SCALING_RETENTION_TIME,
        help_text="Gridwidth factor for retention time.",
    )


    # denoising advanced parameters
    gaussian_filter_sigma = forms.IntegerField(
        label="Gausian Filter Sigma",
        required=False,
        min_value=1,
        max_value=5,
        initial=1,
        help_text="Defines the std deviation of the gaussian kernel.",
    )

    savitzky_golay_filter_window_length = forms.IntegerField(
        label="Savitzky Golay Filter Window Length",
        required=False,
        min_value=3,
        max_value=15,
        initial=9,
        help_text="Length of the filter window.",
    )

    savitzky_golay_filter_poly_order = forms.IntegerField(
        label="Savitzky Golay Filter Polynomial Order",
        required=False,
        min_value=1,
        max_value=10,
        initial=2,
        help_text="The order of the polynomial used to fit the samples.",
    )

    median_filter_kernel_size = forms.IntegerField(
        label="MEDIAN_FILTER kernel size",
        required=False,
        min_value=3,
        max_value=21,
        initial=9,
        help_text="Size of the kernel applied in the filter.",
    )

    # noise_subtraction_1k0_cutoff = forms.FloatField(
    #     label="NOISE_SUBTRACTION X cutoff value",
    #     required=False,
    #     min_value=0.01,
    #     max_value=0.7,
    #     initial=0.4,
    #     help_text="Cutoff value inverse reduced ion mobility. Values smaller than cutoff will be removed, e.g. pre-RIP.",
    # )
    #
    crop_inverse_ion_mobility = forms.FloatField(
        label="CROP_INVERSE_REDUCED_MOBILITY X cutoff value",
        required=False,
        min_value=0.01,
        max_value=0.7,
        initial=0.4,
        help_text="Cutoff value inverse reduced ion mobility. Values smaller than cutoff will be removed, e.g. pre-RIP.",
    )

    discrete_wavelet_transformation_level_irm = forms.IntegerField(
        label="DISCRETE_WAVELET_TRANSFORMATION decomposition level X-axis.",
        required=False,
        min_value=1,
        max_value=8,
        initial=4,
        help_text="Decomposition level of the db8-wavelet for inverse reduced ion mobility.",
    )


    discrete_wavelet_transformation_level_rt = forms.IntegerField(
        label="DISCRETE_WAVELET_TRANSFORMATION decomposition level Y-axis.",
        required=False,
        min_value=1,
        max_value=8,
        initial=2,
        help_text="Decomposition level of the db8-wavelet for retention time.",
    )

    def __init__(self, *args, **kwargs):
        super(ProcessingStepsForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()

        self.helper.layout = Layout(
            TabHolder(
                # make sure to use most recent crispy_forms library version https://github.com/django-crispy-forms/django-crispy-forms/issues/698
                # li should have nav-item and a nav-link
                Tab('Simple',

                    ),

                Tab('Basic',
                    Fieldset('Peak Detection Methods',
                             Field('peak_detection'), ),
                    Fieldset('Peak Alignment Methods',
                             Field('peak_alignment'),
                             ),
                    Fieldset('Normalization / Denoising Methods',
                             Field('denoising_method'),
                             ),
                    ),
                Tab('Advanced',
                    Fieldset('Peak Detection Methods',
                             Fieldset('TOPHAT',
                                      Field('tophat_noise_threshold'),
                                      ),
                             Fieldset('WATERSHED',
                                      Field('watershed_noise_threshold'),
                                      ),
                             Fieldset('JIBB',
                                      Field('jibb_noise_threshold'),
                                      Field('jibb_range_ivr'),
                                      Field('jibb_range_rt'),
                                      ),
                             # VISUALNOW is automatically detected if suffix matches - no need to specify here
                             # Fieldset('VISUALNOW',
                             #          Field('visulanow_file'),
                             #          ),
                             ),
                    Fieldset('Peak Alignment Methods',
                             Fieldset('PROBE_CLUSTERING',
                                  Field('probe_clustering_stepwidth_irm'),
                                  Field('probe_clustering_scaling_rt'),
                              ),
                             Fieldset('DBSCAN',
                                  Field('dbscan_eps'),
                                  Field('dbscan_no_sample'),
                              ),
                     ),
                    Fieldset('Normalization / Denoising Methods',
                        Field('gaussian_filter_sigma'),
                        Field('savitzky_golay_filter_window_length'),
                        Field('savitzky_golay_filter_poly_order'),
                        Field('median_filter_kernel_size'),
                        Field('crop_inverse_ion_mobility'),
                        Field('discrete_wavelet_transformation_level_irm'),
                        Field('discrete_wavelet_transformation_level_rt'),
                     ),
                )
            ),
        )

        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Continue'))

    def get_default_fill(self):
        """
        Return Peak Detection Methods, Peak Alignment Methods and Denoising Method Selection
        :return:
        """
        pdms = [self._coerce_peak_detection(pdmn) for pdmn in self.fields['peak_detection'].initial]
        pam = PeakAlignmentMethod(self.fields['peak_alignment'].initial)
        pdnms = [self._coerce_denoising_method(pdnm) for pdnm in self.fields['denoising_method'].initial]

        return pdms, pam, pdnms


class ProcessingStepsFormMatcher(object):

    def __init__(self, processingStepsForm, peak_layer_filename="", peax_binary_path=""):
        # match parameters
        assert isinstance(processingStepsForm, ProcessingStepsForm)

        preprocessing_parameters = dict()

        selected_peak_detection_methods = processingStepsForm.cleaned_data['peak_detection']
        selected_peak_alignment_method = processingStepsForm.cleaned_data['peak_alignment']
        selected_denoising_method = processingStepsForm.cleaned_data['denoising_method']

        # even if no params chosen, all selected options should be passed
        for method in chain(selected_peak_detection_methods, [selected_peak_alignment_method], selected_denoising_method):
            preprocessing_parameters[method.name] = {}

        # only pass advanced params, if basic option has been selected - keep it clean
        if PeakDetectionMethod.TOPHAT in selected_peak_detection_methods:
            th = dict()
            th['noise_threshold'] = processingStepsForm.cleaned_data['tophat_noise_threshold']
            preprocessing_parameters[PeakDetectionMethod.TOPHAT.name] = th

        if PeakDetectionMethod.WATERSHED in selected_peak_detection_methods:
            ws = dict()
            ws['noise_threshold'] = processingStepsForm.cleaned_data['watershed_noise_threshold']
            preprocessing_parameters[PeakDetectionMethod.WATERSHED.name] = ws

        if (PeakDetectionMethod.VISUALNOWLAYER in selected_peak_detection_methods) and peak_layer_filename:
            vsl = dict()
            vsl['visualnow_filename'] = peak_layer_filename
            preprocessing_parameters[PeakDetectionMethod.VISUALNOWLAYER.name] = vsl

        if ExternalPeakDetectionMethod.PEAX in selected_peak_detection_methods:
            px = dict()
            px['peax_binary_path'] = peax_binary_path
            # create temp dir for raw_file export and peax results
            tmp_dir = os.path.join(tempfile.gettempdir(), '.breath/peax_raw/{}/'.format(hash(os.times())))
            os.makedirs(tmp_dir)
            px['tempdir'] = tmp_dir
            preprocessing_parameters[ExternalPeakDetectionMethod.PEAX.name] = px

        if PeakDetectionMethod.JIBB in selected_peak_detection_methods:
            jibb = dict()
            jibb['noise_threshold'] = processingStepsForm.cleaned_data['jibb_noise_threshold']
            jibb['range_ivr'] = processingStepsForm.cleaned_data['jibb_range_ivr']
            jibb['range_rt'] = processingStepsForm.cleaned_data['jibb_range_rt']
            preprocessing_parameters[PeakDetectionMethod.JIBB.name] = jibb

        if PeakAlignmentMethod.PROBE_CLUSTERING == selected_peak_alignment_method:
            pc = dict()
            pc['threshold_inverse_reduced_mobility'] = processingStepsForm.cleaned_data['probe_clustering_stepwidth_irm']
            pc['threshold_scaling_retention_time'] = processingStepsForm.cleaned_data['probe_clustering_scaling_rt']
            preprocessing_parameters[PeakAlignmentMethod.PROBE_CLUSTERING.name] = pc

        elif PeakAlignmentMethod.DB_SCAN_CLUSTERING == selected_peak_alignment_method:
            dbs = dict()
            dbs['eps'] = processingStepsForm.cleaned_data['dbscan_eps']
            dbs['min_samples'] = processingStepsForm.cleaned_data['dbscan_no_sample']
            preprocessing_parameters[PeakAlignmentMethod.DB_SCAN_CLUSTERING.name] = dbs

        if DenoisingMethod.GAUSSIAN_FILTER in selected_denoising_method:
            gf = dict()
            gf['sigma'] = processingStepsForm.cleaned_data['gaussian_filter_sigma']
            preprocessing_parameters[DenoisingMethod.GAUSSIAN_FILTER.name] = gf

        if DenoisingMethod.SAVITZKY_GOLAY_FILTER in selected_denoising_method:
            sgf = dict()
            sgf['window_length'] = processingStepsForm.cleaned_data['savitzky_golay_filter_window_length']
            sgf['poly_order'] = processingStepsForm.cleaned_data['savitzky_golay_filter_poly_order']
            preprocessing_parameters[DenoisingMethod.SAVITZKY_GOLAY_FILTER.name] = sgf

        if DenoisingMethod.MEDIAN_FILTER in selected_denoising_method:
            mf = dict()
            mf['kernel_size'] = processingStepsForm.cleaned_data['median_filter_kernel_size']
            preprocessing_parameters[DenoisingMethod.MEDIAN_FILTER.name] = mf

        if DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY in selected_denoising_method:
            cirm = dict()
            cirm['cutoff_1ko_axis'] = processingStepsForm.cleaned_data['crop_inverse_ion_mobility']
            preprocessing_parameters[DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY.name] = cirm

        if DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION in selected_denoising_method:
            dwt = dict()
            dwt['level_inverse_reduced_mobility'] = processingStepsForm.cleaned_data['discrete_wavelet_transformation_level_irm']
            dwt['level_retention_time'] = processingStepsForm.cleaned_data['discrete_wavelet_transformation_level_rt']
            preprocessing_parameters[DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION.name] = dwt

        if not peak_layer_filename and PeakDetectionMethod.VISUALNOWLAYER in selected_peak_detection_methods:
            # remove from preprocessing options, as it cant be executed if no layer present
            preprocessing_parameters.pop(PeakDetectionMethod.VISUALNOWLAYER.name)

        self.preprocessing_parameters = preprocessing_parameters
        print(preprocessing_parameters)


    @staticmethod
    def match_with_defaults(peak_layer_filename="", peax_binary_path=""):

        preprocessing_parameters = dict()

        processingStepsForm = ProcessingStepsForm()
        selected_peak_detection_methods, selected_peak_alignment_method, selected_denoising_method = processingStepsForm.get_default_fill()

        # even if no params chosen, all selected options should be passed
        for method in chain(selected_peak_detection_methods, [selected_peak_alignment_method],
                            selected_denoising_method):
            preprocessing_parameters[method.name] = {}

        # only pass advanced params, if basic option has been selected - keep it clean and convenient
        if PeakDetectionMethod.TOPHAT in selected_peak_detection_methods:
            th = dict()
            th['noise_threshold'] = processingStepsForm.fields['tophat_noise_threshold'].initial
            preprocessing_parameters[PeakDetectionMethod.TOPHAT.name] = th

        if PeakDetectionMethod.WATERSHED in selected_peak_detection_methods:
            ws = dict()
            ws['noise_threshold'] = processingStepsForm.fields['watershed_noise_threshold'].initial
            preprocessing_parameters[PeakDetectionMethod.WATERSHED.name] = ws
        if PeakDetectionMethod.VISUALNOWLAYER in selected_peak_detection_methods and peak_layer_filename:
            vsl = dict()
            vsl['visualnow_filename'] = peak_layer_filename
            preprocessing_parameters[PeakDetectionMethod.VISUALNOWLAYER.name] = vsl
        if ExternalPeakDetectionMethod.PEAX in selected_peak_detection_methods:
            px = dict()
            px['peax_binary_path'] = peax_binary_path
            # create temp dir for raw_file export and peax results
            tmp_dir = os.path.join(tempfile.gettempdir(), '.breath/peax_raw/{}/'.format(hash(os.times())))
            os.makedirs(tmp_dir)
            px['tempdir'] = tmp_dir
            preprocessing_parameters[ExternalPeakDetectionMethod.PEAX.name] = px

        if PeakDetectionMethod.JIBB in selected_peak_detection_methods:
            jibb = dict()
            jibb['noise_threshold'] = processingStepsForm.fields['jibb_noise_threshold'].initial
            jibb['range_ivr'] = processingStepsForm.fields['jibb_range_ivr'].initial
            jibb['range_rt'] = processingStepsForm.fields['jibb_range_rt'].initial
            preprocessing_parameters[PeakDetectionMethod.JIBB.name] = jibb

        if PeakAlignmentMethod.PROBE_CLUSTERING == selected_peak_alignment_method:
            pc = dict()
            pc['threshold_inverse_reduced_mobility'] = processingStepsForm.fields[
                'probe_clustering_stepwidth_irm'].initial
            pc['threshold_scaling_retention_time'] = processingStepsForm.fields['probe_clustering_scaling_rt'].initial
            preprocessing_parameters[PeakAlignmentMethod.PROBE_CLUSTERING.name] = pc

        elif PeakAlignmentMethod.DB_SCAN_CLUSTERING == selected_peak_alignment_method:
            dbs = dict()
            dbs['eps'] = processingStepsForm.fields['dbscan_eps'].initial
            dbs['min_samples'] = processingStepsForm.fields['dbscan_no_sample'].initial
            preprocessing_parameters[PeakAlignmentMethod.DB_SCAN_CLUSTERING.name] = dbs

        if DenoisingMethod.GAUSSIAN_FILTER in selected_denoising_method:
            gf = dict()
            gf['sigma'] = processingStepsForm.fields['gaussian_filter_sigma'].initial
            preprocessing_parameters[DenoisingMethod.GAUSSIAN_FILTER.name] = gf

        if DenoisingMethod.SAVITZKY_GOLAY_FILTER in selected_denoising_method:
            sgf = dict()
            sgf['window_length'] = processingStepsForm.fields['savitzky_golay_filter_window_length'].initial
            sgf['poly_order'] = processingStepsForm.fields['savitzky_golay_filter_poly_order'].initial
            preprocessing_parameters[DenoisingMethod.SAVITZKY_GOLAY_FILTER.name] = sgf

        if DenoisingMethod.MEDIAN_FILTER in selected_denoising_method:
            mf = dict()
            mf['kernel_size'] = processingStepsForm.fields['median_filter_kernel_size'].initial
            preprocessing_parameters[DenoisingMethod.MEDIAN_FILTER.name] = mf

        if DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY in selected_denoising_method:
            cirm = dict()
            cirm['cutoff_1ko_axis'] = processingStepsForm.fields['crop_inverse_ion_mobility'].initial
            preprocessing_parameters[DenoisingMethod.CROP_INVERSE_REDUCED_MOBILITY.name] = cirm

        if DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION in selected_denoising_method:
            dwt = dict()
            dwt['level_inverse_reduced_mobility'] = processingStepsForm.fields[
                'discrete_wavelet_transformation_level_irm'].initial
            dwt['level_retention_time'] = processingStepsForm.fields[
                'discrete_wavelet_transformation_level_rt'].initial
            preprocessing_parameters[DenoisingMethod.DISCRETE_WAVELET_TRANSFORMATION.name] = dwt

        if not peak_layer_filename and PeakDetectionMethod.VISUALNOWLAYER in selected_peak_detection_methods:
            # remove from preprocessing options, as it cant be executed if no layer present
            preprocessing_parameters.pop(PeakDetectionMethod.VISUALNOWLAYER.name)

        return preprocessing_parameters


class AnalysisFormMatcher(object):

    def __init__(self, analysisForm):
        # match parameters
        # both forms have the same fields - but gcms form has additional peak alignment option
        assert (isinstance(analysisForm, AnalysisForm) or isinstance(analysisForm, GCMSEvaluationForm))

        performance_measure_parameters = dict()

        # selected_performance_measures = analysisForm.cleaned_data['performance_measure']
        selected_performance_measures = analysisForm.cleaned_data['performance_measure']
        selected_feature_reduction = analysisForm.cleaned_data['feature_reduction']
        # changed, RFC is always enabled, as needed for prediction
        if PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION not in selected_performance_measures:
            selected_performance_measures.append(PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION)

        if PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION in selected_performance_measures:
            rf = dict()
            key_prefix = "random_forest_"
            rf['n_splits_cross_validation'] = analysisForm.cleaned_data[key_prefix+'n_splits_cross_validation']
            rf['n_estimators_random_forest'] = analysisForm.cleaned_data[key_prefix+'n_estimators_random_forest']
            rf['n_of_features'] = analysisForm.cleaned_data[key_prefix+'n_of_features']
            performance_measure_parameters[PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name] = rf

        if PerformanceMeasure.FDR_CORRECTED_P_VALUE in selected_performance_measures:
            fdr_correction = dict()
            key_prefix = "fdr_corrected_p_values_"
            fdr_correction['benjamini_hochberg_alpha'] = analysisForm.cleaned_data[key_prefix+'benjamini_hochberg_alpha']
            fdr_correction['n_of_features'] = analysisForm.cleaned_data[key_prefix+'n_of_features']
            # fdr_correction['n_permutations'] = analysisForm.cleaned_data[key_prefix+'n_permutations']
            performance_measure_parameters[PerformanceMeasure.FDR_CORRECTED_P_VALUE.name] = fdr_correction

        # set params for feature reduction
        if FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES in selected_feature_reduction:
            remove_percentage = dict()
            key_prefix = "remove_percentage_"
            remove_percentage['noise_threshold'] = 1.0 / analysisForm.cleaned_data[key_prefix+'noise_threshold']
            # changed from 100 / val to actual correct computation of val/100
            remove_percentage['percentage_threshold'] = analysisForm.cleaned_data[key_prefix+'percentage_threshold'] / 100.
            performance_measure_parameters[FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES.name] = remove_percentage

        #  make sure decision tree is always built, cant be turned of, so always pass along and parse form fields
        decision_tree = dict()
        key_prefix = "decision_tree_"
        decision_tree['max_depth'] = analysisForm.cleaned_data[key_prefix + 'max_depth']
        decision_tree['min_samples_split'] = analysisForm.cleaned_data[key_prefix + 'min_samples_split']
        decision_tree['min_samples_leaf'] = analysisForm.cleaned_data[key_prefix + 'min_samples_leaf']
        performance_measure_parameters[PerformanceMeasure.DECISION_TREE_TRAINING.name] = decision_tree

        # print(performance_measure_parameters)
        self.performance_measure_parameters = performance_measure_parameters

    @staticmethod
    def get_custom_preprocessing_evaluation_options(min_occurence_per_class):
        analysisForm = AnalysisForm(minimum_occurences_class_label=min_occurence_per_class)
        # get useful number of cv splits
        num_cv_splits = analysisForm.fields['random_forest_n_splits_cross_validation'].initial

        # update number of cv splits to something reasonable
        preprop_parameters, performance_measure_parameters = construct_custom_processing_evaluation_dict(min_num_cv=num_cv_splits)

        # should already be json fiendly
        # json_friendly_params = {k.name : v for k,v in performance_measure_parameters.items()}
        return preprop_parameters, performance_measure_parameters


class ReviewAutoForm(forms.Form):

    def __init__(self, *args, **kwargs):
        super(ReviewAutoForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(

            FormActions(
                Submit('cancel', 'Back', css_class='btn btn-lg btn-dark'),
                Submit('continue', 'Continue', css_class="btn-lg")
            ),
        )

class CrispyReviewForm(forms.Form):

    def __init__(self, *args, **kwargs):
        super(CrispyReviewForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(

            FormActions(
                Submit('dataset', 'Back to dataset selection', css_class="btn-lg btn-info"),
                Submit('processing_steps', 'Back to parameter selection', css_class='btn btn-lg btn-secondary'),
                Submit('continue', 'Continue', css_class="btn-lg btn-success"),
                ),
        )

class AnalysisForm(forms.Form):

    performance_measure = forms.TypedMultipleChoiceField(
        label = "Evaluation Parameters",
        # choices = zip(options_to_name((PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION, PerformanceMeasure.FDR_CORRECTED_P_VALUE)),
        #               options_to_name((PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION, PerformanceMeasure.FDR_CORRECTED_P_VALUE))),
        choices = zip(options_to_name([PerformanceMeasure.FDR_CORRECTED_P_VALUE]),
                                      options_to_name([PerformanceMeasure.FDR_CORRECTED_P_VALUE])),
        coerce = lambda x: PerformanceMeasure(x),
        widget = forms.CheckboxSelectMultiple,
        initial = [PerformanceMeasure.FDR_CORRECTED_P_VALUE.name],
        required=False,  # cant be required and only with a single option
    )
    feature_reduction = forms.TypedMultipleChoiceField(
        label = "Feature Reduction",
        choices = zip(options_to_name(MccImsAnalysis.AVAILABLE_FEATURE_REDUCTION_METHODS), options_to_name(MccImsAnalysis.AVAILABLE_FEATURE_REDUCTION_METHODS)),
        coerce = lambda x: FeatureReductionMethod(x),
        widget = forms.CheckboxSelectMultiple,
        # feature reduction applied by default - may lead to problems
        initial = FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES.name,
        required = True,
    )

    remove_percentage_percentage_threshold = forms.IntegerField(
        label="Percentage threshold for feature reduction",
        required=False,
        min_value=10,
        max_value=100,
        initial=50,
        help_text="Please select a number between 30 and 100",
    )

    remove_percentage_noise_threshold = forms.IntegerField(
        label="Noise threshold for feature reduction",
        required=False,
        min_value=10,
        max_value=100000,
        initial=10000,
        help_text="Select the relative value (1/ (n fold maximum intensity)) below which everything is considered noise.",
    )

    decision_tree_min_samples_leaf = forms.IntegerField(
        label="Minimum number of samples required to be a leaf node.",
        required=False,
        min_value=1,
        max_value=100,
        initial=1,
        help_text="A split is only performed if it leaves at least min_samples_leaf samples in each of the branches.",
    )

    decision_tree_min_samples_split = forms.IntegerField(
        label="Minimum number of samples required to split an internal node.",
        required=False,
        min_value=2,
        max_value=100,
        initial=2,
        help_text="If less samples in current branch, then node will be a leaf.",
    )

    decision_tree_max_depth = forms.IntegerField(
        label="Maximum depth of the decision tree.",
        required=False,
        min_value=2,
        max_value=100,
        initial=5,
        help_text="Only splits nodes until depth is reached. Can lead to impure leave nodes.",
    )

    random_forest_n_estimators_random_forest = forms.IntegerField(
        label="Number of estimators used in the random forest",
        required=False,
        min_value=50,
        max_value=2000,
        initial=2000,
        help_text="Select the number of estimators for the random forest.",
    )

    random_forest_n_of_features = forms.IntegerField(
        label="Number of features reported from the prediction model.",
        required=False,
        min_value=5,
        max_value=1000,
        initial=10,
        help_text="Select the number of features reported from the prediction model.",
    )

    random_forest_n_splits_cross_validation = forms.IntegerField(
        label="Number of splits for cross validation",
        required=False,
        min_value=2,
        max_value=10,
        initial=3,
        help_text="Select the number of splits for the cross validation.",
    )

    fdr_corrected_p_values_benjamini_hochberg_alpha = forms.FloatField(
        label="Alpha for benjamini-hochberg procedure",
        required=False,
        min_value=0.05,
        max_value=0.5,
        initial=0.05,
        help_text="Select confidence interval for false discovery rate.",
    )
    fdr_corrected_p_values_n_of_features = forms.IntegerField(
        label="Number of features reported from the prediction model.",
        required=False,
        min_value=5,
        max_value=1000,
        initial=10,
        help_text="Select the number of features reported from the prediction model.",
    )
    # fdr_corrected_p_values_n_permutations = forms.IntegerField(
    #     label="Number of permutations for multiple testing correction.",
    #     required=False,
    #     min_value=100,
    #     max_value=10000,
    #     initial=1000,
    #     help_text="Select number of permutations for permutation test.",
    # )


    def __init__(self, minimum_occurences_class_label, *args, **kwargs):
        super(AnalysisForm, self).__init__(*args, **kwargs)
        max_splits = min(minimum_occurences_class_label, 10)
        initial_splits = AnalysisForm.get_decent_split_num(minimum_occurence=minimum_occurences_class_label)
        print(f"setting max split to {max_splits}; initial split to {initial_splits}")

        self.fields['random_forest_n_splits_cross_validation'].max_value = max_splits
        self.fields['random_forest_n_splits_cross_validation'].initial = initial_splits

        self.helper = FormHelper()

        self.helper.layout = Layout(
            TabHolder(
                # if tabs are not properly rendered make sure to get the correct version,
                # see https://github.com/django-crispy-forms/django-crispy-forms/issues/698
                # li should have nav-item and a nav-link
                Tab('Simple',),
                Tab('Basic Selection',
                    Fieldset('Evaluation Parameters',
                             Field('performance_measure'),),
                    Fieldset('Feature Reduction',
                             Field('feature_reduction'),
                             ),
                ),
                Tab('Advanced',
                    Fieldset('Evaluation Parameters',
                        Fieldset('Decision Tree',
                             Field('decision_tree_max_depth'),
                             Field('decision_tree_min_samples_split'),
                             Field('decision_tree_min_samples_leaf'),
                             ),
                        Fieldset('Random Forest',
                             Field('random_forest_n_splits_cross_validation'),
                             Field('random_forest_n_estimators_random_forest'),
                             Field('random_forest_n_of_features'),
                             ),
                         Fieldset('FDR corrected p_values',
                             Field('fdr_corrected_p_values_benjamini_hochberg_alpha'),
                             Field('fdr_corrected_p_values_n_of_features'),
                             # Field('fdr_corrected_p_values_n_permutations'),
                             ),
                     ),
                    Fieldset('Feature Reduction',
                             # Field('percentage_threshold'),
                             # Field('percentage_threshold', template="custom-slider.html"),
                             AppendedText('remove_percentage_percentage_threshold', '<span class="input-group-text">%</span>'),
                             Field('remove_percentage_noise_threshold'),
                    ),
                )
            ),

        )

        # self.helper.form_id = 'id-analysisForm'
        # self.helper.form_class = 'blueForms'
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Continue'))

    @staticmethod
    def get_decent_split_num(minimum_occurence):
        # if more than 10 measurements for smallest class - set to 2 fold
        # --> at least 5 per class
        if minimum_occurence >= 25:  # approx 50 or more samples -> 10 fold cross val
            initial_splits = 10
        else:
            initial_splits = minimum_occurence // 5
        # minimum value for cross val is 2
        return max(2,initial_splits)

class PredictionForm(forms.Form):
    """
    Needs to handle selection of prediction models and upload of zipfile
    """
    # based on https://stackoverflow.com/questions/4789466/populate-a-django-form-with-data-from-database-in-view#answer-4789516
    based_on_performance_measure = forms.MultipleChoiceField(
        required=True,
        label="Available Prediction Models",
        widget=forms.CheckboxSelectMultiple,
    )

    based_on_predefined_sets = forms.ModelChoiceField(
        required=False,
        label="Predefined Datasets",
        queryset=PredefinedFileset.objects.all()
    )

    zip_file = forms.FileField(
        required=False,
        validators=[ZipFileValidator(max_size=2000 * 1024 * 1024, check_class_labels=False, check_layer_file=False)],
        label="Zip file",
        help_text="\nPlease upload a zip archive containing the MCC-IMS-measurements to predict. Optionally add a class_labels.csv file to the archive.",
    )



    def __init__(self, web_prediction_model_key, *args, **kwargs):
        super(PredictionForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()

        # need to get possible prediction models -> keys in predictionModel
        pdm_names = [pdm_name for pdm_name in
                     WebPredictionModel.objects.get(pk=web_prediction_model_key).feature_names_by_pdm.keys()]
        # self.available_pdms = pdm_names
        prediction_model_descriptions = [f"Prediction Model based on  {pdm} results" for pdm in pdm_names]

        self.fields['based_on_performance_measure'].choices = zip(pdm_names, prediction_model_descriptions)
        self.fields['based_on_performance_measure'].initial = pdm_names[0]

        # self.fields['web_prediction_model_field'].queryset =
        # self.fields['web_prediction_model_field'].empty_label = None
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Submit'))


    def clean(self):
        from .models import ClassPredictionFileSet

        # based_on_performance_measure = self.cleaned_data.get('based_on_performance_measure') # no need to clean
        based_on_predefined_sets = self.cleaned_data.get('based_on_predefined_sets')
        user_file = self.cleaned_data.get('zip_file')
        # zip_file = None
        if user_file:
            classPredictionFileSet = ClassPredictionFileSet(upload=user_file)
            classPredictionFileSet.save()
            prediction_file_set_id = classPredictionFileSet.pk
        elif based_on_predefined_sets and isinstance(based_on_predefined_sets, PredefinedFileset):
            # zip_file = self.construct_zip_file(based_on_predefined_sets)
            classPredictionFileSet = ClassPredictionFileSet(upload=based_on_predefined_sets.upload)
            classPredictionFileSet.save()
            prediction_file_set_id = classPredictionFileSet.pk
        else:
            msg = ValidationError("Please select an available dataset or upload your own.")
            # self.add_error(based_on_predefined_sets, msg)
            self.add_error(user_file, msg)
            prediction_file_set_id = 0

        if prediction_file_set_id:
            self.cleaned_data['prediction_file_set_id'] = prediction_file_set_id
        return self.cleaned_data


class CustomDetectionAnalysisForm(forms.Form):
    """
    Needs to handle upload of zipfile with class labels and peak detection results
    """

    user_file = forms.FileField(
        required=False,
        validators=[ZipFileValidator(max_size=2000 * 1024 * 1024, check_class_labels=True, check_layer_file=False, check_peak_detection_results=True)],
        label="Your Peak Detection Results",
        help_text="\nPlease upload a zip archive containing the peak detection results or feature matrix for further analysis.",
    )

    based_on_predefined_sets = forms.ModelChoiceField(
        required=False,
        label="Available Datasets",
        queryset=PredefinedCustomPeakDetectionFileSet.objects.all()
    )

    peak_alignment = forms.TypedChoiceField(
        help_text="Select peak alignment method.",
        label="Peak Alignment",
        choices=zip(options_to_name(MccImsAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS),
            options_to_name(MccImsAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS)),
        coerce=lambda x: PeakAlignmentMethod(x),
        initial=PeakAlignmentMethod.PROBE_CLUSTERING.name,
        required=True,
    )

    # Peak Alignment advanced params
    dbscan_eps = forms.FloatField(
        label="DBSCAN eps",
        required=False,
        min_value=0.01,
        max_value=0.5,
        initial=0.025,
        help_text="Distance from the core sample, above this considered outlier.",
    )

    dbscan_no_sample = forms.IntegerField(
        label="DBSCAN #Min Samples",
        required=False,
        min_value=1,
        max_value=50,
        initial=2,
        help_text="Minimum number of samples required for a cluster.",
    )

    probe_clustering_stepwidth_irm = forms.FloatField(
        label="PROBE_CLUSTERING Stepwidth X",
        required=False,
        min_value=0.001,
        max_value=0.1,
        initial=MccImsAnalysis.BASIC_THRESHOLD_INVERSE_REDUCED_MOBILITY,
        help_text="Gridwidth for inverse reduced ion mobility.",
    )

    probe_clustering_scaling_rt = forms.FloatField(
        label="PROBE_CLUSTERING Scaling Y",
        required=False,
        min_value=0.01,
        max_value=0.5,
        initial=MccImsAnalysis.BASIC_THRESHOLD_SCALING_RETENTION_TIME,
        help_text="Gridwidth factor for retention time.",
    )

    def __init__(self, *args, **kwargs):
        super(CustomDetectionAnalysisForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()

        self.helper = FormHelper()
        self.helper.layout = Layout(
            TabHolder(
                # make sure to use most recent crispy_forms library version https://github.com/django-crispy-forms/django-crispy-forms/issues/698
                # li should have nav-item and a nav-link
                Tab('Simple',
                    Fieldset('Dataset',
                             Field('user_file'),
                             Field('based_on_predefined_sets'),
                             ),
                    ),
                Tab('Advanced',
                    Fieldset('Peak Alignment Methods',
                             Field('peak_alignment'),
                             ),
                    Fieldset('PROBE_CLUSTERING',
                             Field('probe_clustering_stepwidth_irm'),
                             Field('probe_clustering_scaling_rt'),
                             ),
                    Fieldset('DBSCAN',
                             Field('dbscan_eps'),
                             Field('dbscan_no_sample'),
                             ),
                    ),
                 ),
            )
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Submit'))


    def clean(self):
        # from .models import ClassPredictionFileSet
        from .models import construct_custom_peak_detection_fileset_from_zip, construct_custom_feature_matrix_from_zip
        based_on_predefined_sets = self.cleaned_data.get('based_on_predefined_sets')

        # from .models import WebPeakDetectionResult
        my_user_file = self.cleaned_data.get('user_file')
        # zip_file = None
        if my_user_file:
            # CustomPeakDetectionFileSet - contains pdr and class label file
            # ids of pdr saved in custom construct

            # Store to disk
            if isinstance(my_user_file, InMemoryUploadedFile):

                filename = my_user_file.name  # received file name
                file_obj = my_user_file
                target_path = default_storage.path('tmp/')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)

                with default_storage.open('tmp/' + filename, 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)

                zip_file_path = Path(target_path).joinpath(filename)
            else:
                zip_file_path = my_user_file.temporary_file_path()

            # check whether pdrs or feature matrix - priority to feature matrix
            if CustomDetectionAnalysisForm.does_contain_feature_matrix(zip_file_path):
                feature_matrix_id = construct_custom_feature_matrix_from_zip(zip_file_path)
                self.cleaned_data['feature_matrix_id'] = feature_matrix_id

            else:
                customPDFS_id = construct_custom_peak_detection_fileset_from_zip(zip_file_path)
                self.cleaned_data['customPDFS_id'] = customPDFS_id

        elif based_on_predefined_sets and isinstance(based_on_predefined_sets, PredefinedCustomPeakDetectionFileSet):
            zip_file_path = based_on_predefined_sets.upload.path

            # does not contain feature matrix - but let's keep it in for completeness sake
            if CustomDetectionAnalysisForm.does_contain_feature_matrix(zip_file_path):
                feature_matrix_id = construct_custom_feature_matrix_from_zip(zip_file_path)
                self.cleaned_data['feature_matrix_id'] = feature_matrix_id

            else:
                customPDFS_id = construct_custom_peak_detection_fileset_from_zip(zip_file_path)
                self.cleaned_data['customPDFS_id'] = customPDFS_id

        else:
            error_strings = ["Could not parse your peak detection results.",]
            for err in self.errors.values():
                # remove "* " representation from error
                error_strings.append(err.as_text()[2:])

            msg = ValidationError(error_strings)
            # self.add_error(based_on_predefined_sets, msg)
            # only non-field errors are rendered by crispy forms - weird enough but we need to give feedback to the users
            # need to raise error
            raise ValidationError(msg)

        # pass chosen peak_alignment to cleaned data
        preprocessing_parameters = dict()
        selected_peak_alignment_method = self.cleaned_data['peak_alignment']
        preprocessing_parameters[selected_peak_alignment_method.name] = {}

        if PeakAlignmentMethod.PROBE_CLUSTERING == selected_peak_alignment_method:
            pc = dict()
            pc['threshold_inverse_reduced_mobility'] = self.cleaned_data['probe_clustering_stepwidth_irm']
            pc['threshold_scaling_retention_time'] = self.cleaned_data['probe_clustering_scaling_rt']
            preprocessing_parameters[PeakAlignmentMethod.PROBE_CLUSTERING.name] = pc

        elif PeakAlignmentMethod.DB_SCAN_CLUSTERING == selected_peak_alignment_method:
            dbs = dict()
            dbs['eps'] = self.cleaned_data['dbscan_eps']
            dbs['min_samples'] = self.cleaned_data['dbscan_no_sample']
            preprocessing_parameters[PeakAlignmentMethod.DB_SCAN_CLUSTERING.name] = dbs

        self.cleaned_data['preprocessing_parameters'] = preprocessing_parameters
        return self.cleaned_data

    @staticmethod
    def does_contain_feature_matrix(zip_path, fm_suffix="_feature_matrix.csv"):
        """
        Check whether contains feature matrix - if yes it has priority further downstream
        :param zip_path:
        :return:
        """
        fm_fn = ""
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = [filename for filename in archive.namelist() if str.endswith(filename, fm_suffix)]
        if fm_fn:
            return True
        else:
            return False


class UploadUserDatasetForm(forms.Form):
    """
    Handle creation process of dataset for website from zipfile with class labels, feature matrix or raw files
    2GB limit for user file - raw files can be huge even if compressed
    """
    user_file = forms.FileField(
        required=False,
        validators=[ZipFileValidator(max_size=2000 * 1024 * 1024, check_class_labels=True,
                                     check_layer_file=True,
                                     check_peak_detection_results=True, check_gcms_raw=True,
                                     )],
        label="Your Dataset",
        help_text="\nPlease upload a zip archive containing the class_labels file, raw files or feature matrix for further analysis.",
    )

    analysis_type = forms.TypedChoiceField(
        required=True,
        label="Type of Dataset",
        choices=AnalysisType.choices(),
        # coerce=lambda x: GCMSProcessingForm._coerce_peak_detection(x),
        # initial=[GCMSPeakDetectionMethod.CENTROIDED],
    )

    train_val_ratio = forms.FloatField(
        required=True,
        initial=.80,  # would be approximate to 5 fold split
        min_value=.05,
        max_value=.95,
    )

    def clean(self):
        # TODO test me
        #  can also raise validation errors from here - doesnt need to be in Validator with access to single field
        # need user context for validation - so create model in view

        my_user_file = self.cleaned_data.get('user_file')
        if my_user_file:

            # Store to disk
            if isinstance(my_user_file, InMemoryUploadedFile):

                filename = my_user_file.name  # received file name
                file_obj = my_user_file
                target_path = default_storage.path('tmp/')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)

                with default_storage.open('tmp/' + filename, 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)

                zip_file_path = Path(target_path).joinpath(filename)
            else:
                zip_file_path = my_user_file.temporary_file_path()

            # check whether split ratio is possible

            try:
                with zipfile.ZipFile(zip_file_path) as tmp_archvie:
                    class_label_fn = MccImsAnalysis.guess_class_label_extension("", tmp_archvie.namelist())
                    class_label_dict = MccImsAnalysis.parse_class_labels_from_ZipFile(zip_file_path, class_label_fn)
                    split_labels_ratio(class_label_dict, self.split_ratio)
            except ValueError:
                raise ValidationError("Bad split ratio. Need at least one sample per class, both in training and validation set.")

        return self.cleaned_data

    @staticmethod
    def does_contain_feature_matrix(zip_path, fm_suffix="_feature_matrix.csv"):
        """
        Check whether contains feature matrix - if yes it has priority further downstream
        :param zip_path:
        :return:
        """
        fm_fn = ""
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = [filename for filename in archive.namelist() if str.endswith(filename, fm_suffix)]
        if fm_fn:
            return True
        else:
            return False



class CustomPredictionForm(forms.Form):
    """
    Needs to handle upload of zipfile with class labels and peak detection results
    """


    based_on_performance_measure = forms.MultipleChoiceField(
        required=True,
        label="Available Prediction Models from training. Classify new samples using:",
        widget=forms.CheckboxSelectMultiple,
    )

    based_on_predefined_sets = forms.ModelChoiceField(
        required=False,
        label="Predefined Datasets",
        queryset=PredefinedCustomPeakDetectionFileSet.objects.all()
    )


    user_file = forms.FileField(
        required=False,
        validators=[ZipFileValidator(max_size=2000 * 1024 * 1024, check_class_labels=True, check_layer_file=False, check_peak_detection_results=True)],
        label="Your Peak Detection Results for prediction.",
        help_text="\nPlease upload a zip archive containing the peak detection results for further analysis. Optionally add a class_labels.csv file to the archive.",
    )

    # ORDER of the arguments is extremely important - can't have a default value before *args - will fail miserably without error
    def __init__(self, web_prediction_model_key, *args, training_model_description="", **kwargs):
        super(CustomPredictionForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()

        # need to get possible prediction models -> keys in predictionModel
        pdm_names = [pdm_name for pdm_name in
                     WebPredictionModel.objects.get(pk=web_prediction_model_key).feature_names_by_pdm.keys()]

        prediction_model_descriptions = [f"Prediction Model based on  {pdm} results" for pdm in pdm_names]

        self.fields['based_on_performance_measure'].choices = zip(pdm_names, prediction_model_descriptions)
        self.fields['based_on_performance_measure'].initial = pdm_names[0]

        # pdm_name = "CUSTOM"

        training_model_description = "Description here"
        if training_model_description:
            self.helper.layout = Layout(
                 Field('based_on_performance_measure'),
                 # HTML(f"<small class='form-text text-muted'>{training_model_description}</small>"),
                 Field('based_on_predefined_sets'),
                 Field('user_file'),
                         )
        else:
            self.helper.layout = Layout(
                Field('based_on_performance_measure'),
                Field('based_on_predefined_sets'),
                Field('user_file'),
            )

        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Submit'))

    def clean(self):
        from .models import construct_custom_peak_detection_fileset_from_zip, construct_custom_feature_matrix_from_zip

        my_user_file = self.cleaned_data.get('user_file')
        based_on_predefined_sets = self.cleaned_data.get('based_on_predefined_sets')
        # zip_file = None
        if my_user_file:
            # CustomPeakDetectionFileSet - contains pdr and class label file
            # ids of pdr saved in custom construct

            if isinstance(my_user_file, InMemoryUploadedFile):

                filename = my_user_file.name  # received file name
                file_obj = my_user_file
                target_path = default_storage.path('tmp/')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)

                with default_storage.open('tmp/' + filename, 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)

                zip_file_path = Path(target_path).joinpath(filename)
            else:
                zip_file_path = my_user_file.temporary_file_path()

            if CustomDetectionAnalysisForm.does_contain_feature_matrix(zip_file_path):
                feature_matrix_id = construct_custom_feature_matrix_from_zip(zip_file_path)
                self.cleaned_data['feature_matrix_id'] = feature_matrix_id

            else:
                customPDFS_id = construct_custom_peak_detection_fileset_from_zip(zip_file_path)
                self.cleaned_data['customPDFS_id'] = customPDFS_id


        elif based_on_predefined_sets and isinstance(based_on_predefined_sets, PredefinedCustomPeakDetectionFileSet):
            zip_file_path = based_on_predefined_sets.upload.path

            # does not contain feature matrix - but let's keep it in for completeness sake
            if CustomDetectionAnalysisForm.does_contain_feature_matrix(zip_file_path):
                feature_matrix_id = construct_custom_feature_matrix_from_zip(zip_file_path)
                self.cleaned_data['feature_matrix_id'] = feature_matrix_id

            else:
                customPDFS_id = construct_custom_peak_detection_fileset_from_zip(zip_file_path)
                self.cleaned_data['customPDFS_id'] = customPDFS_id

        else:
            error_strings = ["Could not parse your peak detection results.",]
            for err in self.errors.values():
                # remove "* " representation from error
                error_strings.append(err.as_text()[2:])

            msg = ValidationError(error_strings)
            # self.add_error(based_on_predefined_sets, msg)
            # only non-field errors are rendered by crispy forms - weird enough but we need to give feedback to the users
            # need to raise error
            raise ValidationError(msg)

        return self.cleaned_data



class GCMSAnalysisForm(forms.Form):
    """
    Needs to handle upload of zipfile with class labels and featureXMLS and mzML mzXML files
    """

    user_file = forms.FileField(
        required=False,
        # 1GB max size - very lenient
        validators=[ZipFileValidator(max_size=2000 * 1024 * 1024, check_class_labels=True,
                check_layer_file=False, check_peak_detection_results=False,
                check_gcms_raw=True, check_gcms_feautures=True,
                )],
        label="Your raw/centroided mzML files or featureXML files.",
        help_text="\nPlease upload a zip archive containing your raw/centroided mzML files or featureXML files for further analysis.",
    )

    based_on_predefined_sets = forms.ModelChoiceField(
        required=False,
        label="Available Datasets",
        queryset=GCMSPredefinedPeakDetectionFileSet.objects.all()
    )

    # added feature matrix support
    # added example files


    def __init__(self, *args, **kwargs):
        super(GCMSAnalysisForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()

        # pdm_name = "CUSTOM"

        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Submit'))


    def clean(self):
        # from .models import ClassPredictionFileSet
        from .models import construct_custom_peak_detection_fileset_from_zip, construct_custom_feature_matrix_from_zip
        from.models import CustomPeakDetectionFileSet

        based_on_predefined_sets = self.cleaned_data.get('based_on_predefined_sets')

        my_user_file = self.cleaned_data.get('user_file')
        if my_user_file:
            # GCMSPeakDetectionFileSet - contains pdr and class label file
            # ids of pdr saved in custom construct

            # Store to disk
            if isinstance(my_user_file, InMemoryUploadedFile):

                filename = my_user_file.name  # received file name
                file_obj = my_user_file
                target_path = default_storage.path('tmp/')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)

                with default_storage.open('tmp/' + filename, 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)

                zip_file_path = Path(target_path).joinpath(filename)
            else:
                zip_file_path = my_user_file.temporary_file_path()
        elif based_on_predefined_sets and isinstance(based_on_predefined_sets, GCMSPredefinedPeakDetectionFileSet):
            zip_file_path = based_on_predefined_sets.upload.path
        else:
            error_strings = ["Could not parse the the zip archive. No matching gcms file endings detected.",]
            for err in self.errors.values():
                # remove "* " representation from error
                error_strings.append(err.as_text()[2:])

            msg = ValidationError(error_strings)
            # self.add_error(based_on_predefined_sets, msg)
            # only non-field errors are rendered by crispy forms - weird enough but we need to give feedback to the users
            # need to raise error
            raise ValidationError(msg)

        # check whether feature matrix, featureXML or rawFiles - priority to feature matrix
        if GCMSAnalysisForm.does_contain_feature_matrix(zip_file_path):
            feature_matrix_id = construct_custom_feature_matrix_from_zip(zip_file_path)
            self.cleaned_data['feature_matrix_id'] = feature_matrix_id

        elif GCMSAnalysisForm.does_contain_featurexml_files(zip_file_path):
            gcms_fileset_id = create_gcms_fileset_from_zip(zip_file_path, use_raw_files=False)
            self.cleaned_data['gcms_fileset_id'] = gcms_fileset_id

        elif GCMSAnalysisForm.does_contain_raw_files(zip_file_path):
            # create GCMSFileSet
            gcms_fileset_id = create_gcms_fileset_from_zip(zip_file_path)
            self.cleaned_data['gcms_fileset_id'] = gcms_fileset_id
        else:

            error_strings = ["Could not parse the the zip archive. No matching gcms file endings detected.", ]
            for err in self.errors.values():
                # remove "* " representation from error
                error_strings.append(err.as_text()[2:])

            msg = ValidationError(error_strings)
            # self.add_error(based_on_predefined_sets, msg)
            # only non-field errors are rendered by crispy forms - weird enough but we need to give feedback to the users
            # need to raise error
            raise ValidationError(msg)

        return self.cleaned_data

    @staticmethod
    def does_contain_feature_matrix(zip_path, fm_suffix="_feature_matrix.csv"):
        """
        Check whether contains feature matrix - if yes it has priority further downstream
        :param zip_path:
        :return:
        """
        fm_fn = []
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = [filename for filename in archive.namelist() if str.endswith(filename, fm_suffix)]
        if fm_fn:
            return True
        else:
            return False

    @staticmethod
    def does_contain_featurexml_files(zip_path):
        """
        Check whether contains raw featureXML files - has priority over raw files
        :param zip_path:
        :return:
        """
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = filter_feature_xmls(dir="", name_list=archive.namelist())
        return fm_fn

    @staticmethod
    def does_contain_raw_files(zip_path):
        """
        Check whether contains raw mzML mzXML files - has lowest priority
        :param zip_path:
        :return:
        """
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = filter_mzml_or_mzxml_filenames(dir="", filelis=archive.namelist())
        return fm_fn



class GCMSProcessingForm(forms.Form):
    """
    Allow user to select between different peak detection methods
    """
    PEAK_DETECTION_OPTIONS = options_to_name(GCMSAnalysis.AVAILABLE_PEAK_DETECTION_METHODS)

    PDO = []
    if GCMSPeakDetectionMethod.CUSTOM.name in PEAK_DETECTION_OPTIONS:
        PDO = [pdo for pdo in PEAK_DETECTION_OPTIONS if pdo != GCMSPeakDetectionMethod.CUSTOM.name]
    PEAK_DETECTION_OPTIONS = PDO

    @staticmethod
    def _coerce_peak_detection(str):
        return GCMSPeakDetectionMethod(str)
    #
    # @staticmethod
    # def _coerce_gcsm_datatype(str):
    #     return GCMSSupportedDatatype(str)
    #
    # # BASIC OPTIONS
    # gcms_datatype = forms.TypedChoiceField(
    #     help_text="Select your gcms Datatype.",
    #     label="GCMS Datatype",
    #     choices=zip(PEAK_DETECTION_OPTIONS, PEAK_DETECTION_OPTIONS),
    #     coerce=lambda x: GCMSProcessingForm._coerce_peak_detection(x),
    #     widget=forms.CheckboxInput,
    #     initial=[GCMSPeakDetectionMethod.CENTROIDED],
    #     required=True,
    # )

    # BASIC OPTIONS
    peak_detection = forms.TypedChoiceField(
        help_text="Select one peak detection method. For centroided data chose centroided, for raw choose isotopewavelet.\n" +
                  "For high-resolution data run your own feature extraction and upload your .featureXML files at " +
                  "<a href='../select_dataset_gcms'>selectdataset_gcms</a>.", # not ideal but works in comparison to reverse
        label="GCMS Peak Detection - must match your data",
        choices=zip(PEAK_DETECTION_OPTIONS, PEAK_DETECTION_OPTIONS),
        coerce=lambda x: GCMSProcessingForm._coerce_peak_detection(x),
        initial=[GCMSPeakDetectionMethod.CENTROIDED],
        required=True,
        )


    def __init__(self, *args, **kwargs):
        super(GCMSProcessingForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()

        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Continue'))

    def get_default_fill(self):
        """
        Return Peak Detection Methods, Peak Alignment Methods and Denoising Method Selection
        :return:
        """
        pdms = [self._coerce_peak_detection(pdmn) for pdmn in self.fields['peak_detection'].initial]
        return pdms


class GCMSProcessingFormMatcher(object):

    def __init__(self, gcms_processingForm):
        # match parameters
        assert isinstance(gcms_processingForm, GCMSProcessingForm)

        preprocessing_parameters = dict()

        selected_peak_detection_method = gcms_processingForm.cleaned_data['peak_detection']
        preprocessing_parameters[selected_peak_detection_method] = {}

        #
        # if GCMSPeakDetectionMethod.CENTROIDED in selected_peak_detection_methods:
        #     centroided = dict()
        #     centroided[]
        #
        # # only pass advanced params, if basic option has been selected - keep it clean
        # if PeakDetectionMethod.TOPHAT in selected_peak_detection_methods:
        #     th = dict()
        #     th['noise_threshold'] = processingStepsForm.cleaned_data['tophat_noise_threshold']
        #     preprocessing_parameters[PeakDetectionMethod.TOPHAT.name] = th

        self.preprocessing_parameters = preprocessing_parameters
        print(preprocessing_parameters)



class GCMSEvaluationForm(forms.Form):
    """
    Form for getting evaluation parameters and feature reduction for GCMS analysis
    """

    peak_alignment = forms.TypedChoiceField(
        help_text="Select peak alignment method. Applied for raw and featureXML input.",
        label="Peak Alignment - will be applied if required",
        choices=zip(options_to_name(GCMSAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS),
            options_to_name(GCMSAnalysis.AVAILABLE_PEAK_ALIGNMENT_METHODS)),
        coerce=lambda x: GCMSAlignmentMethod(x),
        initial=GCMSAlignmentMethod.POSE_FEATURE_GROUPER,
        required=True,
    )

    performance_measure = forms.TypedMultipleChoiceField(
        label = "Evaluation Parameters",
        choices=zip(options_to_name([PerformanceMeasure.FDR_CORRECTED_P_VALUE]),
                    options_to_name([PerformanceMeasure.FDR_CORRECTED_P_VALUE])),
        coerce=lambda x: PerformanceMeasure(x),
        widget=forms.CheckboxSelectMultiple,
        initial=[PerformanceMeasure.FDR_CORRECTED_P_VALUE.name],
        required=False,  # cant be required and only with a single option
    )

    decision_tree_min_samples_leaf = forms.IntegerField(
        label="Minimum number of samples required to be a leaf node.",
        required=False,
        min_value=1,
        max_value=100,
        initial=1,
        help_text="A split is only performed if it leaves at least min_samples_leaf samples in each of the branches.",
    )

    decision_tree_min_samples_split = forms.IntegerField(
        label="Minimum number of samples required to split an internal node.",
        required=False,
        min_value=2,
        max_value=100,
        initial=2,
        help_text="If less samples in current branch, then node will be a leaf.",
    )

    decision_tree_max_depth = forms.IntegerField(
        label="Maximum depth of the decision tree.",
        required=False,
        min_value=2,
        max_value=100,
        initial=5,
        help_text="Only splits nodes until depth is reached. Can lead to impure leave nodes.",
    )
    feature_reduction = forms.TypedMultipleChoiceField(
        label = "Feature Reduction",
        choices = zip(options_to_name(MccImsAnalysis.AVAILABLE_FEATURE_REDUCTION_METHODS), options_to_name(MccImsAnalysis.AVAILABLE_FEATURE_REDUCTION_METHODS)),
        coerce = lambda x: FeatureReductionMethod(x),
        widget = forms.CheckboxSelectMultiple,
        # feature reduction applied by default - may lead to problems
        initial = FeatureReductionMethod.REMOVE_PERCENTAGE_FEATURES.name,
        required = True,
    )

    remove_percentage_percentage_threshold = forms.IntegerField(
        label="Percentage threshold for feature reduction",
        required=False,
        min_value=10,
        max_value=100,
        initial=50,
        help_text="Please select a number between 30 and 100",
    )

    remove_percentage_noise_threshold = forms.IntegerField(
        label="Noise threshold for feature reduction",
        required=False,
        min_value=10,
        max_value=100000,
        initial=10000,
        help_text="Select the relative value (1/ (n fold maximum intensity)) below which everything is considered noise.",
    )

    random_forest_n_estimators_random_forest = forms.IntegerField(
        label="Number of estimators used in the random forest",
        required=False,
        min_value=50,
        max_value=2000,
        initial=2000,
        help_text="Select the number of estimators for the random forest.",
    )

    random_forest_n_of_features = forms.IntegerField(
        label="Number of features reported from the prediction model.",
        required=False,
        min_value=5,
        max_value=1000,
        initial=50,
        help_text="Select the number of features reported from the prediction model.",
    )

    random_forest_n_splits_cross_validation = forms.IntegerField(
        label="Number of splits for cross validation",
        required=False,
        min_value=2,
        max_value=10,
        initial=5,
        help_text="Select the number of splits for the cross validation.",
    )

    fdr_corrected_p_values_benjamini_hochberg_alpha = forms.FloatField(
        label="Alpha for benjamini-hochberg procedure",
        required=False,
        min_value=0.05,
        max_value=0.5,
        initial=0.05,
        help_text="Select confidence interval for false discovery rate.",
    )
    fdr_corrected_p_values_n_of_features = forms.IntegerField(
        label="Number of features reported from the prediction model.",
        required=False,
        min_value=5,
        max_value=1000,
        initial=50,
        help_text="Select the number of features reported from the prediction model.",
    )
    # fdr_corrected_p_values_n_permutations = forms.IntegerField(
    #     label="Number of permutations for multiple testing correction.",
    #     required=False,
    #     min_value=100,
    #     max_value=10000,
    #     initial=1000,
    #     help_text="Select number of permutations for permutation test.",
    # )


    def __init__(self, minimum_occurences_class_label, *args, **kwargs):
        super(GCMSEvaluationForm, self).__init__(*args, **kwargs)
        max_splits = min(minimum_occurences_class_label, 10)
        initial_splits = self.get_decent_split_num(minimum_occurence=minimum_occurences_class_label)
        print(f"setting max split to {max_splits}; initial split to {initial_splits}")

        self.fields['random_forest_n_splits_cross_validation'].max_value = max_splits
        self.fields['random_forest_n_splits_cross_validation'].initial = initial_splits

        self.helper = FormHelper()

        self.helper.layout = Layout(
            TabHolder(
                # if tabs are not properly rendered make sure to get the correct version,
                # see https://github.com/django-crispy-forms/django-crispy-forms/issues/698
                # li should have nav-item and a nav-link
                Tab('Simple',
                    Fieldset('Peak Alignment',
                             Field('peak_alignment'),),
                ),
                Tab('Basic Selection',
                    Fieldset('Evaluation Parameters',
                             Field('performance_measure'),),
                    Fieldset('Feature Reduction',
                             Field('feature_reduction'),
                             ),
                ),
                Tab('Advanced',
                    Fieldset('Evaluation Parameters',
                         Fieldset('Decision Tree',
                                  Field('decision_tree_max_depth'),
                                  Field('decision_tree_min_samples_split'),
                                  Field('decision_tree_min_samples_leaf'),
                                  ),
                        Fieldset('Random Forest',
                             Field('random_forest_n_splits_cross_validation'),
                             Field('random_forest_n_estimators_random_forest'),
                             Field('random_forest_n_of_features'),
                             ),
                         Fieldset('FDR corrected p_values',
                             Field('fdr_corrected_p_values_benjamini_hochberg_alpha'),
                             Field('fdr_corrected_p_values_n_of_features'),
                             # Field('fdr_corrected_p_values_n_permutations'),
                             ),
                     ),
                    Fieldset('Feature Reduction',
                             # Field('percentage_threshold', template="custom-slider.html"),
                             AppendedText('remove_percentage_percentage_threshold', '<span class="input-group-text">%</span>'),
                             Field('remove_percentage_noise_threshold'),
                    ),
                )
            ),

        )

        # self.helper.form_id = 'id-analysisForm'
        # self.helper.form_class = 'blueForms'
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Continue'))

    @staticmethod
    def get_decent_split_num(minimum_occurence):
        # if more than 10 measurements for smallest class - set to 2 fold
        # --> at least 5 per class
        if minimum_occurence >= 25:  # approx 50 or more samples -> 10 fold cross val
            initial_splits = 10
        else:
            initial_splits = minimum_occurence // 5
        # minimum value for cross val is 2
        return max(2, initial_splits)


class GCMSPredictionForm(forms.Form):
    """
    Needs to handle upload of zipfile with class labels, peak detection results, raw files and feature matrix
    """


    based_on_performance_measure = forms.MultipleChoiceField(
        required=True,
        label="Available Prediction Models from training. Classify new samples using:",
        widget=forms.CheckboxSelectMultiple,
    )

    based_on_predefined_sets = forms.ModelChoiceField(
        required=False,
        label="Predefined Datasets",
        queryset=GCMSPredefinedPeakDetectionFileSet.objects.all()
    )


    user_file = forms.FileField(
        required=False,
        validators=[ZipFileValidator(max_size=2000 * 1024 * 1024, check_class_labels=True, check_layer_file=False, check_gcms_feautures=True, check_gcms_raw=True)],
        label="Your raw/centroided mzML files or featureXML files.",
        help_text="\nPlease upload a zip archive containing your raw/centroided mzML files or featureXML files for further analysis.",
    )

    # ORDER of the arguments is extremely important - can't have a default value before *args - will fail misarbly without error
    def __init__(self, web_prediction_model_key, *args, training_model_description="", **kwargs):
        super(GCMSPredictionForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()

        # need to get possible prediction models -> keys in predictionModel
        pdm_names = [pdm_name for pdm_name in
                     WebPredictionModel.objects.get(pk=web_prediction_model_key).feature_names_by_pdm.keys()]

        # Classify samples with
        prediction_model_descriptions = [f"Prediction Model based on {pdm} results" for pdm in pdm_names]

        self.fields['based_on_performance_measure'].choices = zip(pdm_names, prediction_model_descriptions)
        self.fields['based_on_performance_measure'].initial = pdm_names[0]

        # pdm_name = "CUSTOM"

        if not training_model_description:
            training_model_description = "Description here"
        if training_model_description:
            self.helper.layout = Layout(
                 Field('based_on_performance_measure'),
                 # HTML(f"<small class='form-text text-muted'>{training_model_description}</small>"),
                 Field('based_on_predefined_sets'),
                 Field('user_file'),
                         )
        else:
            self.helper.layout = Layout(
                Field('based_on_performance_measure'),
                Field('based_on_predefined_sets'),
                Field('user_file'),
            )

        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Submit'))

    def clean(self):
        from .models import create_gcms_fileset_from_zip, construct_custom_feature_matrix_from_zip

        based_on_predefined_sets = self.cleaned_data.get('based_on_predefined_sets')

        my_user_file = self.cleaned_data.get('user_file')
        if my_user_file:
            # GCMSPeakDetectionFileSet - contains pdr and class label file
            # ids of pdr saved in custom construct

            # Store to disk
            if isinstance(my_user_file, InMemoryUploadedFile):

                filename = my_user_file.name  # received file name
                file_obj = my_user_file
                target_path = default_storage.path('tmp/')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)

                with default_storage.open('tmp/' + filename, 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)

                zip_file_path = Path(target_path).joinpath(filename)
            else:
                zip_file_path = my_user_file.temporary_file_path()
        elif based_on_predefined_sets and isinstance(based_on_predefined_sets, GCMSPredefinedPeakDetectionFileSet):
            zip_file_path = based_on_predefined_sets.upload.path
        else:
            error_strings = ["Could not parse the the zip archive. No matching gcms file endings detected.", ]
            for err in self.errors.values():
                # remove "* " representation from error
                error_strings.append(err.as_text()[2:])

            msg = ValidationError(error_strings)
            # self.add_error(based_on_predefined_sets, msg)
            # only non-field errors are rendered by crispy forms - weird enough but we need to give feedback to the users
            # need to raise error
            raise ValidationError(msg)

        # check whether feature matrix, featureXML or rawFiles - priority to feature matrix
        if self.does_contain_feature_matrix(zip_file_path):
            feature_matrix_id = construct_custom_feature_matrix_from_zip(zip_file_path)
            self.cleaned_data['feature_matrix_id'] = feature_matrix_id

        elif self.does_contain_featurexml_files(zip_file_path):
            gcms_fileset_id = create_gcms_fileset_from_zip(zip_file_path, use_raw_files=False)
            self.cleaned_data['gcms_fileset_id'] = gcms_fileset_id

        elif self.does_contain_raw_files(zip_file_path):
            # create GCMSFileSet
            gcms_fileset_id = create_gcms_fileset_from_zip(zip_file_path)
            self.cleaned_data['gcms_fileset_id'] = gcms_fileset_id
        else:

            error_strings = ["Could not parse the the zip archive. No matching gcms file endings detected.", ]
            for err in self.errors.values():
                # remove "* " representation from error
                error_strings.append(err.as_text()[2:])

            msg = ValidationError(error_strings)
            # self.add_error(based_on_predefined_sets, msg)
            # only non-field errors are rendered by crispy forms - weird enough but we need to give feedback to the users
            # need to raise error
            raise ValidationError(msg)

        return self.cleaned_data


    @staticmethod
    def does_contain_feature_matrix(zip_path, fm_suffix="_feature_matrix.csv"):
        """
        Check whether contains feature matrix - if yes it has priority further downstream
        :param zip_path:
        :return:
        """
        fm_fn = []
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = [filename for filename in archive.namelist() if str.endswith(filename, fm_suffix)]
        if fm_fn:
            return True
        else:
            return False

    @staticmethod
    def does_contain_featurexml_files(zip_path):
        """
        Check whether contains raw featureXML files - has priority over raw files
        :param zip_path:
        :return:
        """
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = filter_feature_xmls(dir="", name_list=archive.namelist())
        return fm_fn

    @staticmethod
    def does_contain_raw_files(zip_path):
        """
        Check whether contains raw mzML mzXML files - has lowest priority
        :param zip_path:
        :return:
        """
        with zipfile.ZipFile(zip_path) as archive:
            fm_fn = filter_mzml_or_mzxml_filenames(dir="", filelis=archive.namelist())
        return fm_fn