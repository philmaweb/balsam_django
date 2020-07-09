# import pdb
import magic  # installed via package python-magic
import numpy as np
import pandas as pd
from zipfile import ZipFile
from io import BytesIO, StringIO
import os
from pathlib import Path
from shutil import rmtree
import joblib

from collections import Counter, OrderedDict
from itertools import chain
from uuid import uuid4

from django import forms
from django.conf import settings
from django.contrib.postgres.fields import JSONField
from django.contrib.postgres.fields.array import ArrayField
from django.contrib.auth.models import User, Group
from django.contrib.auth import get_user_model, authenticate

from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.files.images import ImageFile
from django.core.files.uploadedfile import InMemoryUploadedFile

from django.db import models
from django.template.defaultfilters import filesizeformat
from django.utils.deconstruct import deconstructible
from django.utils.timezone import now

from breathpy.model.BreathCore import (
                              AnalysisResult,
                              MccImsAnalysis,
                              MccImsMeasurement,
                              PredictionModel,
                              construct_default_parameters,
                              InvalidLayerFileError,
                              construct_default_processing_evaluation_steps,
                              GCMSAnalysis,
)
from breathpy.model.ProcessingMethods import (PeakDetectionMethod,
                              ExternalPeakDetectionMethod,
                              NormalizationMethod,
                              PeakAlignmentMethod,
                              DenoisingMethod,
                              PerformanceMeasure,
                              FeatureReductionMethod,
                              GCMSPeakDetectionMethod,
                              GCMSAlignmentMethod,
                              GCMSPreprocessingMethod,
                                                                    )
from breathpy.model.GCMSTools import (filter_mzml_or_mzxml_filenames)


class TempUserManager(models.Manager):
    username_field = get_user_model().USERNAME_FIELD

    def create_temp_user(self):
        """ Create a temporary user. Return 2-tuple of user and username """
        # user_class = User
        user_name = self.generate_username(User)
        # user_name = self.generate_username()
        user = User.objects.create_user(user_name)
        tmp_user = TempUser(user=user)
        tmp_user.save()
        # self.create(user=user)
        return tmp_user, user_name

    def convert(self, form, temp_user_id):
        """ Convert from a temporary user to a real user. Form is a ModelForm associated with the User class"""
        import django.dispatch
        # use old user!
        # new_user = form.save()

        new_user = TempUser.objects.get(pk=temp_user_id).user
        username = form.cleaned_data.get('username')
        raw_password = form.cleaned_data.get('password1')
        email = form.cleaned_data.get('email')
        firstname = form.cleaned_data.get('first_name')
        lastname = form.cleaned_data.get('last_name')
        new_user.set_password(raw_password)
        new_user.username = username
        new_user.first_name = firstname
        new_user.last_name = lastname
        new_user.email = email
        new_user.save()
        new_user = authenticate(username=username, password=raw_password)
        # remove the temp user instance associated with user

        # self.filter(user=new_user).delete()
        TempUser.objects.filter(user=new_user).delete()
        django.dispatch.Signal(providing_args=['user']).send(self, user=new_user)
        return new_user

    # def generate_username(self):
    def generate_username(self, user_class):
        """ Generates a new username for a user """
        # m = getattr(user_class, 'generate_username')
        max_length = user_class._meta.get_field(self.username_field).max_length
        return uuid4().hex[:max_length]


class TempUser(models.Model):
    """
    Temporary User model, linked to real user
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    objects = TempUserManager()

    def __str__(self):
        return "{} : {}".format(self.user, self.created_at)

    def get_username(self):
        return self.user.get_username()


# is required to make migrations possible, see https://docs.djangoproject.com/en/2.0/topics/migrations/#custom-deconstruct-method
@deconstructible
class ZipFileValidator(object):
    # used from https://stackoverflow.com/questions/20272579/django-validate-file-type-of-uploaded-file
    error_messages = {
        'max_size': ("Ensure this file size is not greater than %(max_size)s."
                     " Your file size is %(size)s."),
        'min_size': ("Ensure this file size is not less than %(min_size)s. "
                     "Your file size is %(size)s."),
        'content_type': "Files of type %(content_type)s are not supported.",
        'min_number_of_ims_csv': "Need at least 2 _ims.csv measurement files. "
                                 "You provided ",
        'min_number_of_pdr_csv': "Need at least 2 _peak_detection_result.csv files or a single file ending in _feature_matrix.csv. "
                                 "You provided ",
        'min_number_of_gcms_raw': "Need at least 2 .mzml or .mzxml files. "
                                 "You provided ",
        'min_number_of_featurexml': "Need at least 2 .featureXML files."
                                 "You provided ",
        'contains_class_label_file': 'Archive does not contain a class_label file.'
        'Please add a proper "class_labels.csv" file to the archive and try again.',
        'class_label_file_doesnt_match': 'class_label file does not match measurements. Make sure your '\
                     '"class_labels.csv" file contains only the measurement names of the ones present and try again.',
        'unequal_number_pdrs': 'Make sure your archive only contains peak detection results of a single peak detection method and try again.',
        'layer_file': "Layer file could not be read. Only .csv or single excel sheet supported.",
        'peak_detection_results': "Could not parse provided peak detection results.",
        'too_many_feature_matrices': "Too many feature matrices provided. Please add only a single feature matrix.",
        'feature_matrix': "Could not parse provided feature matrix.",
        'feature_matrix_shape': "Feature matrix does not contain enough rows for cross validation. Need at least 2 rows (samples) and 1 column (features).",
        'feature_matrix_labels': "Feature matrix rows do not match the order of class labels. Both need to be ordered alphabetically.",
    }

    def __init__(self, max_size=None, min_size=None, check_class_labels=True, check_layer_file=True, check_peak_detection_results=False, check_gcms_raw=False, check_gcms_feautures=False):
        self.max_size = max_size
        self.min_size = min_size
        self.content_type = ('application/zip', )
        self.min_number_of_ims_csv = 2
        self.min_number_of_pdr_csv = 2
        self.min_number_of_gcms_raw = 2
        self.min_number_of_featurexml = 2
        self.check_class_labels = check_class_labels
        self.check_layer_file = check_layer_file
        self.check_peak_detection_results = check_peak_detection_results
        self.check_gcms_raw = check_gcms_raw
        self.check_gcms_features = check_gcms_feautures
        self.contains_class_label_file = True
        self.contains_layer_file = True


    def __call__(self, data):
        if self.max_size is not None and data.size > self.max_size:
            params = {
                'max_size': filesizeformat(self.max_size),
                'size': filesizeformat(data.size),
            }
            raise ValidationError(self.error_messages['max_size'],
                                  'max_size', params)

        if self.min_size is not None and data.size < self.min_size:
            params = {
                'min_size': filesizeformat(self.min_size),
                'size': filesizeformat(data.size)
            }
            raise ValidationError(self.error_messages['min_size'],
                                  'min_size', params)

        if self.content_type:
            content_type = magic.from_buffer(data.read(), mime=True)
            # a valid file can be read again in view or file handler without explicitly seek to 0
            data.seek(0)
            # print('content_type', content_type)

            if content_type not in self.content_type:
                params = {'content_type': content_type}
                raise ValidationError(self.error_messages['content_type'],
                                      'content_type', params)

            # special case when both check_gcms_raw and check_gcms_feautures

            # Validate content of zip file
            with ZipFile(data) as zip_file:
                # extended for usage with peak detection results
                # extended for usage with raw gcms and featureXML
                if self.check_peak_detection_results:
                    # also check for feature matrix
                    feature_matrix_filenames = [filename for filename in zip_file.namelist() if str.endswith(filename, "_feature_matrix.csv")]

                    # prioritize feature matrix - is further in pipeline and skips more steps - multiple feature matrices

                    if feature_matrix_filenames:
                        if len(feature_matrix_filenames) > 1:
                            # problem when parsing multiple feature matrices at once - we group them by peak_detection_method_name - if all have CUSTOM we can't separate them later on
                            raise ValidationError(f"{self.error_messages['too_many_feature_matrices']} Got {len(feature_matrix_filenames)}.")
                    else:
                        # check for peak detection results
                        pdr_csv_filenames = [filename for filename in zip_file.namelist() if
                                             str.endswith(filename, "_peak_detection_result.csv")]

                        if len(pdr_csv_filenames) < self.min_number_of_pdr_csv:
                            # params = {'min_number_of_pdr_csv': len(pdr_csv_filenames)}
                            raise ValidationError(f"{self.error_messages['min_number_of_pdr_csv']} {len(pdr_csv_filenames)}.")
                elif self.check_gcms_features:
                    # check for featureXML files
                    featureXMLS = [fn for fn in zip_file.namelist() if str(fn).lower().endswith(".featurexml")]

                    # special case when both check_gcms_raw and check_gcms_feautures
                    if len(featureXMLS) < self.min_number_of_featurexml and not self.check_gcms_raw:
                        raise ValidationError(f"{self.error_messages['min_number_of_featurexml']} {len(featureXMLS)}.")

                elif self.check_gcms_raw:
                    # check for raw gcms files
                    gcms_raw = filter_mzml_or_mzxml_filenames(dir="", filelis=zip_file.namelist())
                    if len(gcms_raw) < self.min_number_of_gcms_raw:

                        # special case when both check_gcms_raw and check_gcms_feautures
                        if self.check_gcms_raw:
                            # if we reach here featureXMLS are also no present
                            error_str_1 = f"{self.error_messages['min_number_of_featurexml']}."
                            error_str_2 = f"{self.error_messages['min_number_of_gcms_raw']} {len(gcms_raw)}."
                            raise ValidationError(f"{error_str_1} and {error_str_2}")
                        else:
                            raise ValidationError(f"{self.error_messages['min_number_of_gcms_raw']} {len(gcms_raw)}.")
                else:
                    # check ims files
                    csv_filenames = [filename for filename in zip_file.namelist() if str.endswith(filename, "_ims.csv")]
                    if len(csv_filenames) < self.min_number_of_ims_csv:
                        # params = {'min_number_of_ims_csv': len(csv_filenames)}
                        raise ValidationError(f"{self.error_messages['min_number_of_ims_csv']} {len(csv_filenames)}.")

                if self.check_class_labels:
                    # check whether class_label file contained
                    class_label_file_list = [filename for filename in zip_file.namelist() if (str.endswith(filename, "class_labels.txt") or str.endswith(filename, "class_labels.tsv") or str.endswith(filename, "class_labels.csv") )]

                    if not class_label_file_list:
                        params = {'contains_class_label_file': False}
                        raise ValidationError(self.error_messages['contains_class_label_file'],
                                              'contains_class_label_file', params)

                    if self.check_peak_detection_results:

                        if feature_matrix_filenames:
                            # try parsing
                            try:
                                class_label_dict, feature_matrix_df = MccImsAnalysis.read_in_custom_feature_matrix(
                                    feature_matrix_filenames[0], zipfile_handle=zip_file)
                            except ValueError:
                                # if value_error during parsing
                                raise ValidationError(
                                    f"{self.error_messages['feature_matrix']} File: {feature_matrix_filenames[0]}")
                            except KeyError:
                                # if value_error during parsing
                                raise ValidationError(
                                    f"{self.error_messages['feature_matrix']} File: {feature_matrix_filenames[0]}")
                            except IndexError:
                                raise ValidationError(
                                    f"{self.error_messages['feature_matrix']} File: {feature_matrix_filenames[0]}")

                            # check matching indices
                            order_check = np.array(list(class_label_dict.keys())) == feature_matrix_df.index.values
                            order_ok = np.all(order_check)

                            # np.all coerces to scalar - even if it doesnt match ...
                            index_ok = isinstance(order_check, np.ndarray)
                            if not order_ok or not index_ok:
                                raise ValidationError(
                                    f"{self.error_messages['feature_matrix_labels']} File: {feature_matrix_filenames[0]}")
                            has_enough_rows = len(feature_matrix_df.index.values) > self.min_number_of_pdr_csv
                            has_enough_cols = len(feature_matrix_df.columns.values) > 1

                            if not (has_enough_rows and has_enough_cols):
                                raise ValidationError(
                                    f"{self.error_messages['feature_matrix_shape']} File: {feature_matrix_filenames[0]}")

                            # create model objects then

                        else:
                            # coerce peak detection method from zip name

                            # coerced_pdm = coerce_pdm_from_path(Path(data).stem)
                            # class_label_dict, pdrs = MccImsAnalysis.read_in_custom_peak_detection(zip_file, zipfile_handle=zip_file, pdm=coerced_pdm)

                            class_label_dict, pdrs = MccImsAnalysis.read_in_custom_peak_detection(zip_file, zipfile_handle=zip_file)
                            pdr_measurement_names = [pdr.measurement_name for pdr in pdrs]

                            key_names = np.array(list(class_label_dict.keys()))
                            # df = pd.DataFrame.from_dict({"key_names":key_names, "pdr_measurement_names":pdr_measurement_names, "valid":key_names == np.array(pdr_measurement_names)})
                            # print(set(key_names).difference(set(pdr_measurement_names)))

                            if (len(key_names) != len(pdr_measurement_names)):
                                raise ValidationError(self.error_messages['unequal_number_pdrs'],
                                                      'contains_class_label_file',
                                                      params={'contains_class_label_file': False}
                                                      )

                            if not all(key_names == np.array(pdr_measurement_names)):
                                raise ValidationError(self.error_messages['class_label_file_doesnt_match'],
                                                      'contains_class_label_file', params={'contains_class_label_file': False}
                                                      )

            if self.check_layer_file:
                # check whether peak_layer file contained
                potential_layer_file = MccImsAnalysis.check_for_peak_layer_file(data)
                if potential_layer_file:
                    try:
                        if isinstance(data, InMemoryUploadedFile):
                            file_path = data.name
                            MccImsAnalysis.parse_peak_layer("{}/{}".format(file_path, potential_layer_file), memory_file=data)
                        else:
                            file_path = data.temporary_file_path()
                            MccImsAnalysis.parse_peak_layer("{}/{}".format(file_path, potential_layer_file) )

                    # except FileNotFoundError:  # no error, just doesnt contain a layer file
                    except InvalidLayerFileError as ilfe:
                        params = {'contains_layer_file': False}
                        raise ValidationError(str(ilfe), 'contains_layer_file', params)
                    except ValueError:
                        params = {'contains_layer_file': False}
                        raise ValidationError(
                            self.error_messages['contains_layer_file'], 'contains_layer_file', params)

                            # self.error_messages['contains_layer_file'], 'contains_layer_file', params)
                    # class_label_file_list = [filename for filename in zip_file.namelist() if str.endswith(filename, "class_labels.txt")]
                    # if not class_label_file_list:
                    #     params = {'contains_class_label_file': False}
                    #     raise ValidationError(self.error_messages['contains_class_label_file'],
                    #                           'contains_class_label_file', params)
                else:
                    self.contains_layer_file = False

    def __eq__(self, other):
        return isinstance(other, ZipFileValidator)


class FileSet(models.Model):
    name = models.CharField(max_length=100)
    raw_files = ArrayField(models.IntegerField(), default=list)
    processed_files = ArrayField(models.IntegerField(), default=list)
    peak_detection_results = ArrayField(models.IntegerField(), default=list)
    class_label_processed_id_dict = JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)


    def delete(self, *args, **kwargs):
        # delete all raw files
        for rf in RawFile.objects.filter(id__in=self.raw_files):
            # this should be enough - as other filetypes have cascade and depend on this
            rf.delete()
            # RawFile.objects.get(id=rfid).delete()

        # delete all preocessed files
        for pf in ProcessedFile.objects.filter(id__in=self.processed_files):
            pf.delete()
            # ProcessedFile.objects.get(id=pfid).delete()

        # delete all pdrs
        for pdr in WebPeakDetectionResult.objects.filter(id__in=self.peak_detection_results):
            pdr.delete()
            # WebPeakDetectionResult.objects.get(id=pdrid).delete()
        super(FileSet, self).delete(*args, **kwargs)




class PredefinedFileset(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.CharField(max_length=100)
    filename_class_label_dict = JSONField(default=dict)
    upload = models.FileField(upload_to='archives/predefined_raw/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.upload:
            self._setup_filename_class_label_dict()

    def _setup_filename_class_label_dict(self):
        # Validate content of zip file
        with ZipFile(self.upload.path) as zip_file:
            # get the list of files
            zip_content = zip_file.namelist()
        class_label_file_name = MccImsAnalysis.guess_class_label_extension(dir_to_search=self.upload.path, file_list_alternative=zip_content)
        label_dict = MccImsAnalysis.parse_class_labels("{}/{}".format(self.upload.path, class_label_file_name))
        ims_filenames = [filename for filename in zip_content if str.endswith(filename, "_ims.csv")]
        # if '.zip/' in csv_filenames[0]:
        #     csv_filenames = [filename.split('.zip/')[1] for filename in csv_filenames]
        # not_contained = [fn for fn in ims_filenames if fn not in label_dict]
        class_label_mapping = {filename: label_dict[filename] for filename in ims_filenames if filename in label_dict}
        self.class_label_mapping = OrderedDict(sorted(class_label_mapping.items(), key=lambda t: t[0]))
        self.save()


    def __str__(self):
        return "{0} - {2}".format(
            self.name, self.pk, self.description, str(self.uploaded_at),)
        # return "PredefinedFileset {0} - pk {1} - {2} - uploaded at {3}".format(
        #     self.name, self.pk, self.description, str(self.uploaded_at),)

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path:
            os.remove(full_media_path)
        super(PredefinedFileset, self).delete(*args, **kwargs)

class CustomPeakDetectionFileSet(models.Model):
    name = models.CharField(max_length=100)
    peak_detection_results = ArrayField(models.IntegerField(), default=list)
    class_label_processed_id_dict = JSONField(default=dict)

    def get_class_label_processed_id_dict(self):
        return OrderedDict(sorted(self.class_label_processed_id_dict.items(), key=lambda t: t[0]))

    def get_used_peak_detection_method(self):
        """
        Return the peak Detection method of the first PeakDetectionResult in this set - assuming all pdr have same pdm
        :return:
        """
        if len(self.peak_detection_results):
            method_name = UnlinkedWebPeakDetectionResult.objects.get(pk=self.peak_detection_results[0]).peak_detection_method_name
            try:
                method = PeakDetectionMethod(method_name)
                return method
            except ValueError:
                method = ExternalPeakDetectionMethod(method_name)
                return method

        return ExternalPeakDetectionMethod.CUSTOM

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path:
            os.remove(full_media_path)
        super(CustomPeakDetectionFileSet, self).delete(*args, **kwargs)


class PredefinedCustomPeakDetectionFileSet(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=100)
    peak_detection_results = ArrayField(models.IntegerField(), default=list)
    class_label_processed_id_dict = JSONField(default=dict)
    upload = models.FileField(upload_to='archives/predefined_pdr/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def get_class_label_processed_id_dict(self):
        return OrderedDict(sorted(self.class_label_processed_id_dict.items(), key=lambda t: t[0]))

    def __str__(self):
        return f"{self.name} - {self.description}"

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path:
            os.remove(full_media_path)
        super(PredefinedCustomPeakDetectionFileSet, self).delete(*args, **kwargs)


class RawFile(models.Model):
    header = JSONField()
    name = models.CharField(max_length=100)
    label = models.CharField(max_length=30)
    created_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='data/raw/')

    def delete(self, *args, **kwargs):
        try:
            os.remove(os.path.join(settings.MEDIA_ROOT, self.file.name))
        except FileNotFoundError:
            pass
        super(RawFile, self).delete(*args, **kwargs)


class ProcessedFile(models.Model):
    raw_file = models.ForeignKey(RawFile, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    label = models.CharField(max_length=30)
    created_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='data/processed/')


    def delete(self, *args, **kwargs):
        os.remove(os.path.join(settings.MEDIA_ROOT, self.file.name))
        super(ProcessedFile, self).delete(*args, **kwargs)


class ClassPredictionFileSet(models.Model):
    one_mb = 1024 * 1024
    upload = models.FileField(upload_to='archives/%Y/%m/%d/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    filename_class_label_dict = JSONField(null=True)  # holds test labels - optional

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.upload and not self.filename_class_label_dict:
            try:
                self._setup_filename_class_label_dict()
            except FileNotFoundError:
                pass


    def _setup_filename_class_label_dict(self):
        # Validate content of zip file
        with ZipFile(self.upload.path) as zip_file:
            # get the list of files
            zip_content = zip_file.namelist()
        class_label_file_name = [fn for fn in zip_content if fn.split("/")[-1].startswith("class_labels")][0]
        label_dict = MccImsAnalysis.parse_class_labels("{}/{}".format(self.upload.path, class_label_file_name))
        ims_filenames = [filename.split('/')[-1] for filename in zip_content if str.endswith(filename, "_ims.csv")]
        print(f"class_label_file_name {class_label_file_name}")
        print(f"label_dict {label_dict}")
        print(f"ims_filenames {ims_filenames}")
        # if '.zip/' in csv_filenames[0]:
        #     csv_filenames = [filename.split('.zip/')[1] for filename in csv_filenames]
        # not_contained = [fn for fn in ims_filenames if fn not in label_dict]
        self.filename_class_label_dict = {filename: label_dict[filename] for filename in ims_filenames if filename in label_dict}
        self.save()

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path:
            os.remove(full_media_path)
        super(ClassPredictionFileSet, self).delete(*args, **kwargs)


class AnalysisFileSet(models.Model):
    """
    Holds base logic for Analysis and serves as interface for later processing
    """
    # will hold a zipfile and create a directory for extraction of the zip?
    # delete content after 3 months
    one_mb = 1024 * 1024
    # used_validators = models.FileField.validators
    # used_validators.extend([ZipFileValidator(max_size=100*one_mb)])
    # upload = models.FileField(upload_to=user_directory_path,
    upload = models.FileField(upload_to='archives/%Y/%m/%d/',
                              validators=[ZipFileValidator(max_size=2000 * one_mb)])
    uploaded_at = models.DateTimeField(auto_now_add=True)

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # file_set_id = models.ForeignKey(FileSet, on_delete=models.CASCADE)

    def __str__(self):
        return "AnalysisFileSet {0} - {1} - uploaded at {2} by {3}".format(self.pk, self.upload.name, str(self.uploaded_at), self.user)

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path and self.upload.name: #sometimes we have no upload
            os.remove(full_media_path)
        super(AnalysisFileSet, self).delete(*args, **kwargs)


class UnlinkedWebPeakDetectionResult(models.Model):
    """
    A wrapper for Peak Detection Results
    """
    filename = models.CharField(max_length=100)
    peak_detection_method_name = models.CharField(max_length=20)
    csv_file = models.FileField(upload_to='data/peak_detection/')

    @staticmethod
    def fields_from_peak_detection_result(peak_detection_result):
        # check first for required attributes
        if not UnlinkedWebPeakDetectionResult._pdr_attribute_check(peak_detection_result):
            raise ValueError("PeakDetectionresult {} does not have expected attributes.")
        else:
            pdm_name = peak_detection_result.peak_detection_step.name
            result_filename = f"{peak_detection_result.measurement_name}_{pdm_name}" \
                                  f"{peak_detection_result.peak_detection_result_suffix}"
            return result_filename, pdm_name


    @staticmethod
    def _pdr_attribute_check(peak_detection_result):
        return hasattr(peak_detection_result, "peak_detection_step") and \
               hasattr(peak_detection_result, "measurement_name") and \
               hasattr(peak_detection_result, "peak_detection_result_suffix")

    def __str__(self):
        return f"UnlinkedWebPeakDetectionResult {self.pk} - PDMName: {str(self.peak_detection_method_name)}"

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.csv_file.name)
        # don't remove setup media on autoclean - only remove the associated files
        os.remove(full_media_path)
        super(UnlinkedWebPeakDetectionResult, self).delete(*args, **kwargs)


class WebPeakDetectionResult(models.Model):
    """
    A wrapper to manage to connection between measurement file and detected peaks
    """
    preprocessed_measurement = models.ForeignKey(ProcessedFile, on_delete=models.CASCADE)
    filename = models.CharField(max_length=100)
    peak_detection_method_name = models.CharField(max_length=20)
    csv_file = models.FileField(upload_to='data/peak_detection/')

    @staticmethod
    def fields_from_peak_detection_result(peak_detection_result):
        # check first for required attributes
        if not WebPeakDetectionResult._pdr_attribute_check(peak_detection_result):
            raise ValueError("PeakDetectionresult {} does not have expected attributes.")
        else:
            pdm_name = peak_detection_result.peak_detection_step.name
            result_filename = f"{peak_detection_result.measurement_name}_{pdm_name}" \
                f"{peak_detection_result.peak_detection_result_suffix}"
            return result_filename, pdm_name

    @staticmethod
    def _pdr_attribute_check(peak_detection_result):
        return hasattr(peak_detection_result, "peak_detection_step") and \
               hasattr(peak_detection_result, "measurement_name") and \
               hasattr(peak_detection_result, "peak_detection_result_suffix")


    def __str__(self):
        return f"WebPeakDetectionResult {self.pk} - PDMName: {str(self.peak_detection_method_name)}"

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.csv_file.name)
        os.remove(full_media_path)
        super(WebPeakDetectionResult, self).delete(*args, **kwargs)


class WebImsSet(models.Model):
    # will hold a zipfile and create a directory for extraction of the zip?
    # delete content after 30 days
    # see DIMANA cleanupresults.py --> cleans up results after setting.CLEANUP_RESULTS_OLDER_DAYS
    # file will be saved to MEDIA_ROOT/user_<id>/<filename>
    one_mb = 1024 * 1024
    upload = models.FileField(upload_to='archives/%Y/%m/%d/',
                              validators=[ZipFileValidator(max_size=2000 * one_mb)])
    uploaded_at = models.DateTimeField(auto_now_add=True)

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_set = models.ForeignKey(FileSet, on_delete=models.CASCADE)

    def __str__(self):
        return f"WebImsSet {self.pk} - {self.upload.name} - uploaded at {str(self.uploaded_at)} by {self.user}"

    def get_dataset_info(self):
        """
        Create descriptive dictionary from zip archive
        List number of files, class_labels and assignment and visualnow_file
        :return:
        """
        with ZipFile(self.upload.path) as zip_file:
            #     # get the list of files
            # print(zip_file.namelist())
            csv_filenames = []
            class_label_file_list = []
            peak_layer_file_list = []

            for filename in zip_file.namelist():
                if str.endswith(filename, "_ims.csv"):
                    csv_filenames.append(filename.split("/")[-1])
                elif str.startswith(filename.split("/")[-1], "class_labels"):
                    class_label_file_list.append(filename)
                elif (str.endswith(filename, "layer.csv") or str.endswith(filename, "layer.xls")):
                    # peak_layer_file_list.append(filename.split("/")[-1])
                    peak_layer_file_list.append(filename)

            number_of_files = len(csv_filenames)
            class_label_dict = MccImsAnalysis.parse_class_labels(self.upload.path + "/" + class_label_file_list[0])
            if peak_layer_file_list:
                peak_layer_filename = peak_layer_file_list[0]

        # take only tiny_candy.zip as dataset name
        # 'setup/tiny_candy.zip'
        dataset_name = self.upload.name.rsplit("/", maxsplit=1)[1]
        # pre-match csv_filenames with labels in class_label_dict
        matched_tuples = [(fn, class_label_dict.get(fn, '')) for fn in csv_filenames]
        # labeled_measurements = OrderedDict(matched_tuples)
        rv = {"csv_filenames": csv_filenames, "number_of_files": number_of_files, "class_label_dict": class_label_dict,
              "dataset_name": dataset_name, 'labeled_measurements': matched_tuples}
        if peak_layer_file_list:
            rv['peak_layer_filename'] = peak_layer_filename
            # print(peak_layer_filename)
        return rv

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path:
            os.remove(full_media_path)
        super(WebImsSet, self).delete(*args, **kwargs)


class WebCustomSet(AnalysisFileSet):
    # basically list of UnlinkedWebPeakDetectionResult and class label dict
    file_set = models.ForeignKey(CustomPeakDetectionFileSet, on_delete=models.CASCADE)

    def __str__(self):
        return f"WebCustomSet {self.pk} - {self.upload.name} - uploaded at {str(self.uploaded_at)} by {self.user}"


class GCMSPeakDetectionFileSet(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=100)
    peak_detection_results = ArrayField(models.IntegerField(), default=list)
    peak_detection_method_name = models.CharField(max_length=100, default=GCMSPeakDetectionMethod.CUSTOM.name)
    class_label_processed_id_dict = JSONField(default=dict)
    class_label_dict = JSONField(default=dict)
    uploaded_at = models.DateTimeField(auto_now_add=True)  # basically created at for deleting
    has_unlinked_peak_detection_results = models.BooleanField(default=True)

    def get_class_label_processed_id_dict(self):
        return OrderedDict(sorted(self.class_label_processed_id_dict.items(), key=lambda t: t[0]))

    def __str__(self):
        return f"{self.name} - {self.description}"

    def get_peak_detection_result_objects(self):
        """
        Returns queryset to iterate over with linked / unlinked peak detection results
        :return:
        """
        if self.has_unlinked_peak_detection_results:
            return GCMSUnlinkedPeakDetectionResult.objects.filter(id__in=self.peak_detection_results)
        else:
            return GCMSPeakDetectionResult.objects.filter(id__in=self.peak_detection_results)

    def get_peak_detection_result_paths(self):
        """
        Returns path to feture xml files for our convenience
        :return:
        """
        paths = []
        for pdr_object in self.get_peak_detection_result_objects():
            paths.append(Path(pdr_object.file.file.name))  # get absolute path - not path relative to media path
        return paths

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path:
            os.remove(full_media_path)
        super(GCMSPeakDetectionFileSet, self).delete(*args, **kwargs)


class GCMSFileSet(models.Model):
    """
    Object holding references to all files for regular gcms analysis until evaluation where the `MccImsAnalysisWrapper` takes over
    """
    name = models.CharField(max_length=100)
    raw_files = ArrayField(models.IntegerField(), default=list)
    peak_detection_fileset = models.ForeignKey(GCMSPeakDetectionFileSet, on_delete=models.CASCADE, blank=True, null=True)   # for initialization
    class_label_processed_id_dict = JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def delete(self, *args, **kwargs):
        # delete all raw files
        for rf in GCMSRawMeasurement.objects.filter(id__in=self.raw_files):
            # this should be enough - as other filetypes have cascade and depend on this
            rf.delete()

        # delete
        self.peak_detection_fileset.delete()
        super(GCMSFileSet, self).delete(*args, **kwargs)

    def get_class_label_processed_id_dict(self):
        return OrderedDict(sorted(self.class_label_processed_id_dict.items(), key=lambda t: t[0]))


    def get_raw_file_objects(self):
        """
        Return queryset of rawfile objects
        :return:
        """
        return GCMSRawMeasurement.objects.filter(id__in=self.raw_files)

    def get_raw_file_paths(self):
        """
        Return paths of rawfile objects
        :return:
        """
        return [Path(rf.file.name) for rf in self.get_raw_file_objects()]


class MccImsAnalysisWrapper(models.Model):
    """
    A Wrapper around the MCC IMS analysis class
    """
    # Fields
    ims_set = models.ForeignKey(WebImsSet, on_delete=models.CASCADE, null=True) # allows null in DB - in case we have a custom set
    custom_set = models.ForeignKey(WebCustomSet, on_delete=models.CASCADE, blank=True, null=True) # empty in regular usecase - for custom peak detection results
    gcms_set = models.ForeignKey(GCMSFileSet, on_delete=models.CASCADE, blank=True, null=True) # empty in regular usecase - for GCMS file management
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    preprocessing_options = JSONField()  # populated by fields in select Parameters form
    evaluation_options = JSONField()  # populated by fields of evaluation form
    class_label_mapping = JSONField()  # hold dict mapping measurement names to class labels
    is_automatic_analysis = models.BooleanField(default=False)  # automatic approach?
    is_custom_analysis = models.BooleanField(default=False)   # using custom peak detection results?
    is_custom_feature_matrix_analysis = models.BooleanField(default=False)   # using custom feature matrix as input
    is_gcms_analysis = models.BooleanField(default=False)   # using gcms data for analysis?
    automatic_selected_method_name = models.CharField(max_length=100, default="")
    evaluation_model_description = models.CharField(max_length=200, default="")
    created_at = models.DateTimeField(auto_now_add=True)


    @staticmethod
    def _unpack_raw_files_from_zip_prediction(prediction_file_set_id):
        classPredictionFileSet = ClassPredictionFileSet.objects.get(pk=prediction_file_set_id)
        upload_path = classPredictionFileSet.upload.path
        raw_file_ids = []
        measurements = []
        with ZipFile(classPredictionFileSet.upload.path) as archive:
            csv_file_fn_processed_fn_tuples = [("{0}/{1}".format(upload_path, filename), filename, filename.split("/")[-1])
                                  for filename in archive.namelist() if str.endswith(filename, "_ims.csv")]

            class_label = "Test"
            for (csv_file, fn, processed_filename) in csv_file_fn_processed_fn_tuples:
                # class_label = name_label_dict.get(processed_filename, None)
                measurement = MccImsMeasurement(BytesIO(archive.read(fn)), class_label=class_label)
                raw_buff = measurement.export_raw(use_buffer=True)
                raw_file = RawFile(header=measurement.header, name=measurement.filename,
                                   file=ContentFile(
                                       raw_buff, name=measurement.filename[:-4]),
                                   label=measurement.class_label)
                raw_file.save()
                raw_file_ids.append(raw_file.pk)
                measurements.append(measurement)
                # class_label_map[processed_filename] = class_label

        # self.class_label_mapping = class_label_map
        # self.save()
        return raw_file_ids, measurements


    def _unpack_raw_files_from_zip(self):
        webImsSet = WebImsSet.objects.get(pk=self.ims_set.pk)
        raw_file_ids = []
        with ZipFile(webImsSet.upload.path) as archive:
            csv_file_fn_processed_fn_tuples = [("{0}/{1}".format(webImsSet.upload.path, filename), filename, filename.split("/")[-1])
                                  for filename in archive.namelist() if str.endswith(filename, "_ims.csv")]

            name_label_dict = OrderedDict(webImsSet.get_dataset_info()['labeled_measurements'])
            # pdb.set_trace()
            # (csv_file, fn) = csv_file_fn_tuples[0]
            class_label_map = {}
            for (csv_file, fn, processed_filename) in csv_file_fn_processed_fn_tuples:
                class_label = name_label_dict.get(processed_filename, None)
                measurement = MccImsMeasurement(BytesIO(archive.read(fn)), class_label=class_label)
                raw_buff = measurement.export_raw(use_buffer=True)
                raw_file = RawFile(header=measurement.header, name=measurement.filename,
                                   file=ContentFile(
                                       raw_buff, name=measurement.filename[:-4]),
                                   label=measurement.class_label)
                raw_file.save()
                raw_file_ids.append(raw_file.pk)
                class_label_map[processed_filename] = class_label

        self.class_label_mapping = class_label_map
        self.save()
        return raw_file_ids

    def reinitialize_gcms_analysis(self, gcms_fileset_id=0):
        """
        REINITIALIZE ANALYSIS for usage with gcms alignment and regular feature handling
        :return:
        """

        # initialize for Peak detection
        # create gcmsAnalysis object

        # doesnt make sense to initialize gcms analysis - don't need to set anything
        if self.is_custom_feature_matrix_analysis:
            raise ValueError("Use different method, only for featureXML.")

        else:
            # could be linked or unlinked, doesnt matter as long as we dont try to access the raw measurements
            features_xml_paths = self.gcms_set.peak_detection_fileset.get_peak_detection_result_paths()
            # pdr_ids = self.gcms_set.peak_detection_fileset.peak_detection_results
            preprocessing_parameters = self.preprocessing_options  # also included are peak alignment options

            # initialize for peakAlignment + feature evaluation / reduction
            gcms_analysis = GCMSAnalysis(
                    measurements=[],
                    preprocessing_steps=preprocessing_parameters.keys(),
                    preprocessing_parameters=preprocessing_parameters,
                    dataset_name= self.gcms_set.peak_detection_fileset.name,
                    dir_level="",
            )

            gcms_analysis.set_class_label_dict(self.class_label_mapping)
            gcms_analysis.feature_xml_files = features_xml_paths

            if gcms_fileset_id:
                #   if extending feature_xml files with test-set
                #       will need to deconvolute feature matrix and class_label_dict afterwards
                test_gcms_set = GCMSFileSet.objects.get(pk=gcms_fileset_id)
                both_class_label_dicts = self.class_label_mapping
                both_class_label_dicts.update(test_gcms_set.class_label_processed_id_dict)

                # sort united class label dict by featurexml / rawnames
                sorted_both_class_label_dicts = OrderedDict(sorted(both_class_label_dicts.items(), key=lambda t: t[0]))

                # apply same sorting to feature xmls
                test_feature_xml_paths = test_gcms_set.peak_detection_fileset.get_peak_detection_result_paths()
                both_features_xml_paths = features_xml_paths.copy()
                both_features_xml_paths.extend(test_feature_xml_paths)

                # sorting needs to take into account the paths - take stem for that to approximate class_label name
                sorted_both_features_xml_paths = sorted(both_features_xml_paths, key=lambda s: Path(s).stem)

                # set gcms_analysis attributes with updated paths and class labels
                gcms_analysis.set_class_label_dict(sorted_both_class_label_dicts)
                gcms_analysis.feature_xml_files = sorted_both_features_xml_paths


                #   else: - quicker to run - harder to implement
                #       need to create additional alignment function for test_alignment

            plot_params, file_params = construct_default_parameters(self.gcms_set.name, folder_name="", make_plots=True)
            plot_params['use_buffer'] = True


            return gcms_analysis, plot_params, file_params


    def reinitialize_mcc_ims_analysis(self, fileset_id=None, custom_fileset_id=None, single_peak_detection_method_name=""):
        """
        REINITIALIZE ANALYSIS - if fileset_id given load from specific set - if single_peak_detection_method_name is
        given, filter preprocessed files and peak_detection_results by this peak detection method - used in prediction
        if custom_fileset_id is given - prepare from that fileset
        :param fileset_id:
        :param custom_fileset_id:
        :param single_peak_detection_method_name:
        :return:
        """
        custom_fileset = None
        fileset = None

        # get files from db
        if fileset_id is not None:
            fileset = FileSet.objects.get(pk=fileset_id)
            print(f"Using {fileset}")
        elif custom_fileset_id is not None:
            custom_fileset = CustomPeakDetectionFileSet.objects.get(pk=custom_fileset_id)
            print(f"Using {custom_fileset}")
        else:
            # fileset = self.ims_set.file_set_id
            fileset = self.ims_set.file_set
            print(f"Using {fileset}")

        # not used for custom fileset
        # filter by peak_detection method if supplied
        if single_peak_detection_method_name:
            if fileset is not None:
                processed_files = ProcessedFile.objects.filter(id__in=fileset.processed_files, peak_detection_method_name=single_peak_detection_method_name)
                pdr_files = WebPeakDetectionResult.objects.filter(id__in=fileset.peak_detection_results, peak_detection_method_name=single_peak_detection_method_name)
        else:
            if fileset is not None:
                processed_files = ProcessedFile.objects.filter(id__in=fileset.processed_files)
                pdr_files = WebPeakDetectionResult.objects.filter(id__in=fileset.peak_detection_results)

        # don't have processed file if using custom fileset, just peak detection results
        if custom_fileset is not None:
            processed_files = []
            pdr_files = UnlinkedWebPeakDetectionResult.objects.filter(id__in=custom_fileset.peak_detection_results)

        # peak_detection_results are just a list - need to seperate by peak_detection_method and put into dict
        # print(len(pdr_files))
        # for pdr in pdr_files: print(pdr.peak_detection_method)
        # for pdr in pdr_files: print(pdr)

        outfile_names = [pf.name for pf in processed_files]
        # reinitialize analysis object with measurements and peak_detection_results

        if fileset is not None:
            fileset_name = fileset.name
        if custom_fileset is not None:
            fileset_name = custom_fileset.name

        # remove zip form fileset name
        if fileset_name.endswith(".zip"):
            fileset_name = fileset_name[:-4]

        plot_params, file_params = construct_default_parameters(fileset_name, folder_name="", make_plots=True)
        plot_params['use_buffer'] = True

        # print([pf.label for pf in processed_files])
        # print(f"len(processed_files) = {len(processed_files)}")
        # reinitialize analysis with files from db
        measurements = [MccImsMeasurement.import_from_csv(pf.file.path, class_label=pf.label) for pf in processed_files]
        # for m, pf in zip(measurements, processed_files): m.set_class_label(pf.label)

        visualnow_layer_path = ""
        if PeakDetectionMethod.VISUALNOWLAYER.name in self.preprocessing_options:
            if custom_fileset_id:
                visualnow_layer_path = ""
            else:
                visualnow_layer_path = self.preprocessing_options[PeakDetectionMethod.VISUALNOWLAYER.name][
                    'visualnow_filename']


        preprocessing_steps = self.preprocessing_options
        preprocessing_parameters = self.preprocessing_options

        mcc_ims_analysis = MccImsAnalysis(measurements, preprocessing_steps=preprocessing_steps, outfile_names=outfile_names,
                                          preprocessing_parameters=preprocessing_parameters,
                                          dir_level=file_params['dir_level'],
                                          dataset_name=file_params['folder_name'],
                                          class_label_file="",
                                          peax_binary_path=settings.PEAX_BINARY_PATH,
                                          visualnow_layer_file=visualnow_layer_path,
                                          performance_measure_parameters=self.evaluation_options
                                          )

        mcc_ims_analysis.set_class_label_dict(OrderedDict(self.class_label_mapping))

        # need to give concrete fielnames, not pk to import function
        # pdr_filenames = [pdrf.file.path for pdrf in pdr_files]

        # get all peak detection methods from analysis and sort pdr by pdm_name
        pdr_dict = {pdm.name: [] for pdm in mcc_ims_analysis.peak_detection_combined}
        for pdr_file in pdr_files:
            # need to make sure filenames match between assignment in results for different imports - otherwise we'll have issues
            pdr_dict[pdr_file.peak_detection_method_name].append(pdr_file.csv_file.path)

        # reimport peak_detection results
        for pdm_name, lis_of_pdr in pdr_dict.items():
            try:
                pdm = PeakDetectionMethod(pdm_name)
            except ValueError:
                pdm = ExternalPeakDetectionMethod(pdm_name)
                pass
            mcc_ims_analysis.import_results_from_csv_list(lis_of_pdr, pdm)

        # for pdm_name, df in self.dict_of_df.items():
        #     print(pdm_name, df.shape)
        # print(len(measurements))
        return mcc_ims_analysis, plot_params, file_params


    def prepare_custom_fm_approach(self, feature_matrix_model):

        # feature_matrix = feature_matrix_model
        measurements = []
        outfile_names = []
        fileset_name = "custom_feature_matrix"

        preprocessing_steps = self.preprocessing_options
        preprocessing_parameters = self.preprocessing_options

        plot_params, file_params = construct_default_parameters(fileset_name, folder_name="", make_plots=True)
        plot_params['use_buffer'] = True

        mcc_ims_analysis = MccImsAnalysis(measurements, preprocessing_steps=preprocessing_steps,
                                          outfile_names=outfile_names,
                                          preprocessing_parameters=preprocessing_parameters,
                                          dir_level=file_params['dir_level'],
                                          dataset_name=file_params['folder_name'],
                                          class_label_file="",
                                          peax_binary_path=settings.PEAX_BINARY_PATH,
                                          performance_measure_parameters=self.evaluation_options
                                          )
        mcc_ims_analysis.set_class_label_dict(OrderedDict(self.class_label_mapping))

        # try to coerce PeakDetectionMethod - don't just default to CUSTOM
        coerced_pdm = feature_matrix_model.get_used_peak_detection_method()

        fm_dict = {coerced_pdm.name: feature_matrix_model.get_feature_matrix()}
        mcc_ims_analysis.analysis_result = AnalysisResult(peak_alignment_result=None, based_on_measurements=measurements,
                                                          peak_detection_steps=[coerced_pdm],
                                                          peak_alignment_step=None, dataset_name=fileset_name,
                                                          class_label_dict=mcc_ims_analysis.get_class_label_dict(), feature_matrix=fm_dict)

        return mcc_ims_analysis, plot_params, file_params, coerced_pdm


    @staticmethod
    def prepare_test_measurements(zip_path):
        with ZipFile(zip_path) as archive:
            csv_file_fn_tuples = [("{0}/{1}".format(zip_path, filename), filename)
                                  for filename in archive.namelist() if str.endswith(filename, "_ims.csv")]

            measurements = [MccImsMeasurement(BytesIO(archive.read(fn)), class_label=None) for
                                (csv_file, fn) in csv_file_fn_tuples]

        pks = []
        # need to export for peax - save in DB as Raw file
        for measurement in measurements:
            # raw_buff = measurement.export_to_csv(use_buffer=True)
            raw_buff = measurement.export_raw(use_buffer=True)
            raw_file = RawFile(header=measurement.header, name=measurement.filename,
                               file=ContentFile(raw_buff, name=measurement.filename[:-4]),
                               label="Test")
            pks.append(raw_file.save())
            measurement.raw_filename = raw_file.file.path
        return pks, measurements


    @staticmethod
    def create_automatic_analysis(ims_set, user):
        """
        Create a MccImsAnalysisWrapper instance for automatic processing - using default values and all
        available peak detection methods
        :param imsset_pk:
        :param user:
        :return:
        """
        from .forms import ProcessingStepsFormMatcher, AnalysisForm
        dataset_info = ims_set.get_dataset_info()
        peak_layer_filename = dataset_info.get('peak_layer_filename', '')

        if peak_layer_filename:
            absolute_path_visualnow_layer = "{}/{}".format(ims_set.upload.path, peak_layer_filename)
        else:
            absolute_path_visualnow_layer = ''

        # made sure user/platform can select visualnow and it doesnt crash if no layer found
        default_preprocessing_options = ProcessingStepsFormMatcher.match_with_defaults(
            peak_layer_filename=absolute_path_visualnow_layer,
            peax_binary_path=settings.PEAX_BINARY_PATH)



        counter = Counter(dataset_info['class_label_dict'].values())
        minimum_occurences_class_label = sorted(counter.values())[0]
        # make sure cross val split num is decently selected
        initial_splits = AnalysisForm.get_decent_split_num(minimum_occurence=minimum_occurences_class_label)

        default_evaluation_options = construct_default_processing_evaluation_steps(initial_splits)[1]
        # convert them to strings to be json seriazable
        default_evaluation_option_names = {k.name: v for k, v in default_evaluation_options.items()}
        # request.session['preprocessing_parameters'] = default_evaluation_option_names

        ims_model = MccImsAnalysisWrapper(ims_set=ims_set,
                                          preprocessing_options=default_preprocessing_options,
                                          evaluation_options=default_evaluation_option_names,
                                          class_label_mapping={},
                                          user=user,
                                          is_automatic_analysis=True,
                                          )

        ims_model.save()
        return ims_model.pk

    @staticmethod
    def prepare_reduced_fm_json_representation_list(analysis_id):
        fm_full_qs = FeatureMatrix.objects.filter(analysis__pk=analysis_id, is_training_matrix=True)

        # if automatic - could only get the matrices which were considered best
        # but would mean users have lest overview
        reduced_fm_json_representation_list = []

        # one feature matrix per peak detection
        for fm_model in fm_full_qs:
            peak_detection_method_name = fm_model.peak_detection_method_name
            for (r_fm_df, evaluation_method_name) in fm_model.get_reduced_feature_matrices():
                reduced_fm_json_representation_list.append(FeatureMatrix.feature_matrix_df_to_json(
                    r_fm_df, evaluation_method_name=evaluation_method_name,
                    peak_detection_method_name=peak_detection_method_name))
        return reduced_fm_json_representation_list

    def get_dataset_info(self):
        """
        Create descriptive dictionary from zip archive
        List number of files, class_labels and assignment and visualnow_file
        :return:
        """
        # added feature matrix to dict for download
        # if self.is_automatic_analysis:
        #     fm_models = FeatureMatrix.objects.filter(analysis__pk=self.pk, peak_detection_method_name=self.automatic_selected_method_name)
        # else:
        fm_full_qs = FeatureMatrix.objects.filter(analysis__pk=self.pk, is_training_matrix=True)

        # shown training matrix was actually same as test matrix - due to javascript function name collisions - fixed

        fm_json_representation_list = MccImsAnalysisWrapper.prepare_fm_for_template(fm_full_qs)
        pm_json_representation_list = MccImsAnalysisWrapper.prepare_fm_for_template(FeatureMatrix.objects.filter(analysis__pk=self.pk, is_prediction_matrix=True))
        reduced_fm_json_representation_list = MccImsAnalysisWrapper.prepare_reduced_fm_json_representation_list(analysis_id=self.pk)

        if self.is_custom_analysis or self.is_gcms_analysis:
            # custom analysis or gcms - ims set is not set - also don't have archive anymore available
            # get class labels from analysis directly
            class_label_dict = self.class_label_mapping
            number_of_files = len(class_label_dict)

            if self.is_gcms_analysis:
                # get dataset name
                dataset_name = self.gcms_set.name
            else:
                dataset_name = "unknown"
            # pre-match csv_filenames with labels in class_label_dict ,  as required by template
            matched_tuples = [(fn, label) for fn,label in class_label_dict.items()]

            csv_filenames = list(class_label_dict.keys())
            # labeled_measurements = OrderedDict(matched_tuples)
            rv = {"csv_filenames": csv_filenames, "number_of_files": number_of_files,
                  "class_label_dict": class_label_dict,
                  "dataset_name": dataset_name, 'labeled_measurements': matched_tuples,
                  "trainings_matrices": fm_json_representation_list,
                  "reduced_trainings_matrices": reduced_fm_json_representation_list,
                  "prediction_matrices": pm_json_representation_list,
                  }
            return rv

        elif self.ims_set is not None:
            rv = self.ims_set.get_dataset_info()
            rv['trainings_matrices'] = fm_json_representation_list
            rv['prediction_matrices'] = pm_json_representation_list
            rv['reduced_trainings_matrices'] = reduced_fm_json_representation_list
            return rv
        else:
            return dict()

    @staticmethod
    def prepare_fm_for_template(feature_matrix_queryset):
        rv = []
        for tm_object in feature_matrix_queryset:
            trainings_matrix = tm_object.get_feature_matrix()
            peak_ids = trainings_matrix.columns.values
            feature_rows = np.round(trainings_matrix.values, decimals=3)

            display_feature_matrix = True
            # if more than 2000 entries, won't display
            if len(feature_rows) * len(peak_ids) > 2000:
                display_feature_matrix = False

            rv.append({
                "pk": tm_object.pk,
                "peak_detection_method_name": tm_object.peak_detection_method_name,
                "display_feature_matrix": display_feature_matrix,
            })
        return rv

    def prepare_evaluation_stats(self):
        """
        Return formated statistics tables for context and display on site
        Best Features Table (rounded)
        :return:
        """

        best_feature_columns_start = ["performance_measure_name", "peak_detection_method_name", "class_comparison"]
        best_feature_columns_end = ["peak_id", "inverse_reduced_mobility", "radius_inverse_reduced_mobility",
                                    "retention_time", "radius_retention_time"]

        template_best_feature_columns_start = ["Evaluation Method", "Peak Detection Method", "Class Comparison"]

        template_best_feature_columns_end = ["Peak ID", "Inverse Reduced Ion Mobility",
                                             "Radius Inverse Reduced Ion Mobility", "Retention Time",
                                             "Radius Retention Time"]

        # make sure we only use indices that are available -> if only feature matrix font have peak coordinates
        if self.is_custom_feature_matrix_analysis or self.is_gcms_analysis:
            best_feature_columns_end = ['peak_id', ]
            template_best_feature_columns_end = ['Peak ID', ]

        rf_column_list = ["gini_decrease", "corrected_p_values"]
        template_rf_column_list = ["Gini Decrease", "FDR Corrected P Value"]
        fdr_column_list = ["null_hyp_rejected", "raw_p_values", "corrected_p_values"]
        template_fdr_column_list = ["Null Hypothesis Rejected", "Raw P Value", "FDR Corrected P Value"]

        # if feature matrix analysis or gcms analysis we don't have peak coordinates
        if not (self.is_custom_feature_matrix_analysis or self.is_gcms_analysis):
            best_feature_rounding_columns = ["inverse_reduced_mobility", "radius_inverse_reduced_mobility",
                                             "retention_time",
                                             "radius_retention_time"]
            best_feature_rounding_columns_precision = [3, 3, 1, 1]

        rf_rounding_columns = ["gini_decrease", "corrected_p_values"]
        rf_rounding_columns_precision = [3, 12]
        fdr_rounding_columns = ["raw_p_values", "corrected_p_values"]
        fdr_rounding_columns_precision = [12, 12]

        statistics_columns = ["accuracy", "standard_deviation_accuracy", "precision", "standard_deviation_precision",
                              "recall", "area_under_curve_macro", "F1-Score", "standard_deviation_F1-Score"]
        template_statistics_columns = ["Accuracy", "Standard Deviation Accuracy", "Precision",
                                       "Standard Deviation Precision", "Recall", "Area Under Curve (Macro)", "F1-Score",
                                       "Standard Deviation F1-Score"]
        statistics_columns_precision = [3] * len(statistics_columns)

        # stats = get_object_or_404(StatisticsModel, analysis_id=analysis_id).stats_dict

        stats_models = StatisticsModel.objects.all().filter(analysis=self)
        # preformat stats for use in template
        stats_for_template = []

        for stats_model in stats_models:
            stats_by_evaluation = []
            evaluation_method_name = stats_model.evaluation_method_name
            analysis_statistics_per_peak_detection = stats_model.statistics_dict

            # RF performance measure
            if evaluation_method_name == PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name:
                current_columns = rf_column_list
                template_current_columns = template_rf_column_list
                if not (self.is_custom_feature_matrix_analysis or self.is_gcms_analysis):
                    best_features_to_round = best_feature_rounding_columns + rf_rounding_columns
                    best_features_to_round_precision = best_feature_rounding_columns_precision + rf_rounding_columns_precision
                else:
                    best_features_to_round = rf_rounding_columns
                    best_features_to_round_precision = rf_rounding_columns_precision

            # FDR Performance evaluation
            else:
                current_columns = fdr_column_list
                template_current_columns = template_fdr_column_list

                if not (self.is_custom_feature_matrix_analysis or self.is_gcms_analysis):
                    best_features_to_round = best_feature_rounding_columns + fdr_rounding_columns
                    best_features_to_round_precision = best_feature_rounding_columns_precision + fdr_rounding_columns_precision
                else:
                    best_features_to_round = fdr_rounding_columns
                    best_features_to_round_precision = fdr_rounding_columns_precision

            # pvalues - shouldnt be rounded to 0 if too small - but will be if < 1e-12
            # round the columns to certain precision
            best_features_round_dict = {col: prec for col, prec in
                                        zip(best_features_to_round, best_features_to_round_precision)}
            best_features_df = stats_model.get_best_features_df().round(best_features_round_dict)

            # reorder best features columns for tableview
            columns_ordered = best_feature_columns_start + current_columns + best_feature_columns_end

            ordered_best_features_df = best_features_df[columns_ordered]
            template_columns_ordered = template_best_feature_columns_start + template_current_columns + template_best_feature_columns_end

            # best_features_df['peak_detection_method_name']
            for peak_detection_method_name, best_features_group in ordered_best_features_df.groupby(
                    'peak_detection_method_name'):

                if self.is_automatic_analysis and peak_detection_method_name != self.automatic_selected_method_name:
                    continue
                else:

                    # round statistics and put into ordered list
                    ordered_stats = []

                    # we only have a stats dict for cross validation, which happens only for RF
                    # check if cross_validation worked - should at least have a few keys, contain the method name and not an error
                    cross_val_failed = not len(analysis_statistics_per_peak_detection.keys()) or \
                                       peak_detection_method_name not in analysis_statistics_per_peak_detection or \
                                       "error" in analysis_statistics_per_peak_detection[
                                           peak_detection_method_name].keys()
                    cross_val_didnt_fail = not cross_val_failed

                    if evaluation_method_name == PerformanceMeasure.RANDOM_FOREST_CLASSIFICATION.name and cross_val_didnt_fail:

                        orig_stats = analysis_statistics_per_peak_detection[peak_detection_method_name]
                        for stats_key, stats_name, stats_precision in zip(
                                statistics_columns, template_statistics_columns, statistics_columns_precision):
                            ordered_stats.append(np.round(orig_stats[stats_key], stats_precision))
                    # print(ordered_stats)
                    stats_by_evaluation.append({
                        "evaluation_method": evaluation_method_name,
                        "peak_detection_name": peak_detection_method_name,
                        "best_features": best_features_group.values.tolist(),
                        "stats": ordered_stats,
                        "stats_header_names": template_statistics_columns,
                    })
            stats_for_template.append((evaluation_method_name, template_columns_ordered, stats_by_evaluation))
        return stats_for_template

    def __str__(self):
        dataset_name = ""
        if self.ims_set is not None:
            dataset_name = self.ims_set.upload.name
        elif self.custom_set is not None:
            dataset_name = self.custom_set.upload.name

        return f"MccImsAnalysisWrapper {self.pk} - auto {self.is_automatic_analysis} - {dataset_name} - options: {str(self.preprocessing_options)}"

class WebPredictionModel(models.Model):
    """
    A wrapper for the core PredictionModel - dicts mapping peak detection methods to pickles and features -
    its attributes match and can be used to re-initialize
    """
    name = models.CharField(max_length=100)
    scipy_predictor_pickle = models.FileField(upload_to='prediction/prediction_model/')
    feature_names_by_pdm = JSONField()
    mcc_ims_analysis = models.ForeignKey(MccImsAnalysisWrapper, on_delete=models.CASCADE)
    class_labels = JSONField()  # = predicted from model - or as label for prediction_result

    # makes use of its webimssets parameter sets

    def __str__(self):
        pdmns = [pdmn for pdmn in self.feature_names_by_pdm.keys()]
        return f"WebPredictionModel {self.pk} - Analysis: {self.mcc_ims_analysis.pk} - PDMNS: {pdmns}"


    def delete(self, *args, **kwargs):
        os.remove(os.path.join(settings.MEDIA_ROOT, self.scipy_predictor_pickle.name))
        super(WebPredictionModel, self).delete(*args, **kwargs)


class PredictionResult(models.Model):

    web_prediction_model = models.ForeignKey(WebPredictionModel, on_delete=models.CASCADE)
    class_assignment = JSONField()  # maps measurement name to class
    original_class_labels = JSONField(default=dict)  # if available from PredictionFileSet
    peak_detection_method_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "PredictionResult {0} - Analysis: {1} - PDMN: {2}".format(self.pk, self.web_prediction_model.mcc_ims_analysis.pk, self.peak_detection_method_name)


def figure_user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/figures/<filename>
    return f'user_{instance.user.id}/figures/{filename}'

class PlotModel(models.Model):
    # taken from https://stackoverflow.com/questions/35581356/save-matplotlib-plot-image-into-django-model
    # possibly render plots using plotly http://thecuriouscoder.com/create-chart-using-plotly-django/
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    analysis = models.ForeignKey(MccImsAnalysisWrapper, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=300)
    based_on_peak_detection_method_name = models.CharField(max_length=100)
    # figure = models.ImageField(upload_to='figures/', blank=True)
    figure = models.ImageField(upload_to=figure_user_directory_path, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    # to save it create is like so:
    # PlotModel(figure=io_figure)
    class Meta:
        abstract = True

    def __str__(self):
        return f"PlotModel {self.pk} - Analysis {self.analysis.pk} - PDMN: {str(self.based_on_peak_detection_method_name)}"

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.figure.name)
        os.remove(full_media_path)
        super(PlotModel, self).delete(*args, **kwargs)



class ClusterPlotModel(PlotModel):
    based_on_peak_alignment_method_name = models.CharField(max_length=100)
    pass

    def __str__(self):
        return "ClusterPlotModel {0} - Analysis {1} - PDMN: {2}".format(self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name))

class BoxPlotModel(PlotModel):
    based_on_performance_measure_name = models.CharField(max_length=100)
    based_on_peak_id = models.CharField(max_length=100)
    pass

    def __str__(self):
        return "BoxPlotModel {0} - Analysis {1} - PDMN: {2}".format(self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name))

class RocPlotModel(PlotModel):
    based_on_performance_measure_name = models.CharField(max_length=100)
    pass

    def __str__(self):
        return "RocPlotModel {0} - Analysis {1} - PDMN: {2}".format(self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name))

class HeatmapPlotModel(PlotModel):
    # filled with BDXY_YYMMDDHHMM_ims
    based_on_measurement = models.CharField(max_length=100)
    class Meta:
        abstract = True

    def __str__(self):
        return "HeatmapPlotModel {0} - Analysis {1} - PDMN: {2}".format(self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name))

class ClasswiseHeatMapPlotModel(PlotModel):
    based_on_measurement_list = JSONField(default=list)
    class_label = models.CharField(max_length=100)

    def __str__(self):
        return "ClasswiseHeatMapPlotModel {0} - Analysis {1} - PDMN: {2} - Class: {3}".format(
            self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name), str(self.class_label))

class IntensityPlotModel(HeatmapPlotModel):
    pass

    def __str__(self):
        return "IntensityPlotModel {0} - Analysis {1} - PDMN: {2}".format(self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name))

class OverlayPlotModel(HeatmapPlotModel):

    pass

    def __str__(self):
        return "OverlayPlotModel {0} - Analysis {1} - PDMN: {2}".format(self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name))


class BestFeaturesOverlayPlot(HeatmapPlotModel):
    based_on_performance_measure_name = models.CharField(max_length=100)
    pass


    def __str__(self):
        return "BestFeaturesOverlayPlot {0} - Analysis {1} - PDMN: {2} - PerformanceMeasure: {3}".format(
            self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name),
            str(self.based_on_performance_measure_name))



class DecisionTreePlotModel(PlotModel):
    based_on_performance_measure_name = models.CharField(max_length=100)
    pass

    def __str__(self):
        return "DecisionTreePlotModel {0} - Analysis {1} - PDMN: {2} - PerformanceMeasure: {3}".format(
            self.pk, self.analysis.pk, str(self.based_on_peak_detection_method_name), str(self.based_on_performance_measure_name))

class StatisticsModel(models.Model):
    # only here to hold a dictionary
    # web_prediction_model = models.ForeignKey(WebPredictionModel, on_delete=models.CASCADE)
    analysis = models.ForeignKey(MccImsAnalysisWrapper, on_delete=models.CASCADE)
    # user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    evaluation_method_name = models.CharField(max_length=100)
    # statistics_df = models.FileField(upload_to=stats_user_directory_path)
    statistics_dict = JSONField()
    best_features_df = models.FileField(upload_to='data/best_features/')

    def get_best_features_df(self):
        """Deserialize and reimport Dataframe from CSV"""
        # drop full NaN columns
        return pd.read_csv(self.best_features_df.path, index_col=0).dropna(axis=1, how='all')

    def __str__(self):
        return f"StatisticsModel {self.pk} - Analysis {self.analysis.pk} - EvalMethodName: {str(self.evaluation_method_name)}"


class FeatureMatrix(models.Model):
    """
    hold a feature matrix and a reference to an analysis
    """
    # made nullable for usage without analysis - eg when uploading a custom feature matrix or gcms processing
    analysis = models.ForeignKey(MccImsAnalysisWrapper, on_delete=models.CASCADE, blank=True, null=True)
    name = models.CharField(max_length=100)
    peak_detection_method_name = models.CharField(max_length=100)
    file = models.FileField(upload_to='data/feature_matrix/')
    # has to be held somwhere if no analysis yet
    class_label_dict = JSONField(default=dict)  # needs to be sorted before using
    is_training_matrix = models.BooleanField(default=True)  # we only create FeatureMatrix for training
    is_prediction_matrix = models.BooleanField(default=False)  # we only create FeatureMatrix for training

    def get_feature_matrix(self):
        """Deserialize and reimport Dataframe from CSV"""
        return pd.read_csv(self.file.path, index_col=0)

    def __str__(self):
        an_id = "unkonwn"
        if self.analysis is not None:
            an_id = self.analysis.pk
        return f"FeatureMatrix {self.pk} - Analysis {an_id} - PDMName: {str(self.peak_detection_method_name)}"


    def get_class_label_dict(self):
        """
        Return sorted instance
        :return:
        """
        return OrderedDict(sorted(self.class_label_dict.items(), key=lambda t: t[0]))

    def get_used_peak_detection_method(self):
        """
        Return the peak Detection method of the feature matrix
        :return:
        """
        if self.peak_detection_method_name:
            method_name = self.peak_detection_method_name
            try:
                method = PeakDetectionMethod(method_name)
                return method
            except ValueError:
                try:
                    method = ExternalPeakDetectionMethod(method_name)
                    return method

                # also need to care about gcms methods
                except ValueError:
                    try:
                        method = GCMSPeakDetectionMethod(method_name)
                        return method
                    except ValueError:
                        pass
                        # return the CUSTOM method instead

        return ExternalPeakDetectionMethod.CUSTOM


    def get_owning_user(self):
        """
        Get user from analysis
        :return: `User` object
        """
        return self.analysis.user


    def get_reduced_feature_matrices(self):
        """
        returns [tuples] of (reduced_matrix, evaluation_method_name) from statsmodel
        get all statsmodels associated with analysis
          select columns from feature matrix that are used for building prediction model - the reduced / best columns
        """
        stats_model_qs = StatisticsModel.objects.filter(analysis=self.analysis)  # should be two models - one per evaluation method

        # 1 reduced feature matrix for each evaluation option
        rv = []
        full_feature_matrix = self.get_feature_matrix()

        wanted_pdmn = self.peak_detection_method_name

        full_feature_columns = set(full_feature_matrix.columns.values)
        # full_feature_matrix = stats_model_object.get_best_features_df()
        for stats_model_object in stats_model_qs:
            # get feature names from stats_model - prediction model are not always existent when calling this
            # reduced_feature_set = set(stats_model_object.get_best_features_df()['peak_id'].values)

            # filter by peak_detection_method_name from feature matrix and get best peaks from stats model
            master_feature_matrix = stats_model_object.get_best_features_df()
            feature_matrix = master_feature_matrix.loc[master_feature_matrix['peak_detection_method_name'] == wanted_pdmn]

            # sort peak_ids alphabetically - before by performance measure
            best_peak_ids = sorted(feature_matrix['peak_id'].values)

            # could lead to KeyError when not all identical columns names / peak_ids used

            reduced_feature_matrix = full_feature_matrix.loc[full_feature_matrix.index, full_feature_matrix.columns.intersection(best_peak_ids)]

            rv.append((reduced_feature_matrix, stats_model_object.evaluation_method_name))
        return rv



    @staticmethod
    def feature_matrix_df_to_json(dataframe, evaluation_method_name, peak_detection_method_name=ExternalPeakDetectionMethod.CUSTOM.name, pk=0):
        """
        Convert feature_matrix (pandas dataframe) to expected json
        :param dataframe:
        :return:
        """
        measurement_names = [fn for fn in dataframe.index.values]
        peak_ids = dataframe.columns.values
        feature_rows = np.round(dataframe.values, decimals=3)
        if pk:
            return {
                "pk": pk,
                "measurement_names": measurement_names,
                "peak_ids": peak_ids,
                "feature_rows": zip(measurement_names, feature_rows),
                "peak_detection_method_name": peak_detection_method_name,
                "evaluation_method_name": evaluation_method_name,
            }
        else:
            return {
                "measurement_names": measurement_names,
                "peak_ids": peak_ids,
                "feature_rows": zip(measurement_names, feature_rows),
                "peak_detection_method_name": peak_detection_method_name,
                "evaluation_method_name": evaluation_method_name,
            }


    def get_feature_matrix_details_json(self, evaluation_method_name):
        """
        Convert feature_matrix (pandas dataframe) to expected json for view function
        :param dataframe:
        :return:
        """
        fm = self.get_feature_matrix()
        measurement_names = [fn for fn in fm.index.values]
        peak_ids = fm.columns.values
        feature_rows = np.round(fm.values, decimals=3)
        return {
            "measurement_names": measurement_names,
            "peak_ids": peak_ids,
            "feature_rows": zip(measurement_names, feature_rows),
            "peak_detection_method_name": self.peak_detection_method_name,
            "evaluation_method_name": evaluation_method_name,
        }

    def get_feature_matrix_data_json(self):
        """
        Return serialized dataframe - as json
        :param evaluation_method_name:
        :return:
        """

        feature_matrix = self.get_feature_matrix()
        measurement_names = feature_matrix.index.values

        feature_matrix.reset_index()
        feature_matrix['Measurement'] = measurement_names

        # Measurement column was added last - so is at last position
        # rearrange to first column
        cols = feature_matrix.columns.values
        cols = cols[:-1]
        cols = np.insert(cols, 0, "Measurement")

        # requires recent version of pandas - resort with measurement at first col
        result = feature_matrix[cols].to_json(orient="split", index=False)

        # return json
        return result



    # def __unicode__(self):
    #     # returns tuples of matching feature matrices
    #     trainings_matrix = self.get_feature_matrix()
    #     measurement_names = [fn.rsplit("_preprocessed", maxsplit=1)[0] for fn in trainings_matrix.index.values]
    #     peak_ids = trainings_matrix.columns.values
    #     feature_rows = np.round(trainings_matrix.values, decimals=3)
    #     return {"measurement_names": measurement_names,
    #         "peak_ids": peak_ids,
    #         "feature_rows": zip(measurement_names, feature_rows),
    #         "peak_detection_method_name": self.peak_detection_method_name}


    def delete(self, *args, **kwargs):
        os.remove(os.path.join(settings.MEDIA_ROOT, self.file.name))
        super(FeatureMatrix, self).delete(*args, **kwargs)



class GCMSRawMeasurement(models.Model):
    """
    A Model to handle Raw GCMS files
    """

    name = models.CharField(max_length=100)
    label = models.CharField(max_length=30)
    created_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='data/raw_gcms/')

    def delete(self, *args, **kwargs):
        try:
            os.remove(os.path.join(settings.MEDIA_ROOT, self.file.name))
        except FileNotFoundError:
            pass
        super(GCMSRawMeasurement, self).delete(*args, **kwargs)



class GCMSPeakDetectionResult(models.Model):
    """
    A wrapper to manage to connection between raw gcms file and resulting feature xml file
    """
    # do we even need linked PDR?
    raw_measurement = models.ForeignKey(GCMSRawMeasurement, on_delete=models.CASCADE)
    filename = models.CharField(max_length=100)
    file = models.FileField(upload_to='data/peak_detection_gcms/')
    peak_detection_method_name = models.CharField(max_length=20)

    def __str__(self):
        return f"GCMSPeakDetectionResult {self.pk} - PDMName: {str(self.peak_detection_method_name)}"

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.file.name)
        os.remove(full_media_path)
        super(GCMSPeakDetectionResult, self).delete(*args, **kwargs)


class GCMSUnlinkedPeakDetectionResult(models.Model):
    """
    A wrapper to manage to connection between raw gcms file and resulting feature xml file
    """
    filename = models.CharField(max_length=100)
    file = models.FileField(upload_to='data/peak_detection_gcms/')
    peak_detection_method_name = models.CharField(max_length=20)

    def __str__(self):
        return f"GCMSPeakDetectionResult {self.pk} - PDMName: {str(self.peak_detection_method_name)}"

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.file.name)
        os.remove(full_media_path)
        super(GCMSUnlinkedPeakDetectionResult, self).delete(*args, **kwargs)



class GCMSPredefinedPeakDetectionFileSet(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=100)
    peak_detection_results = ArrayField(models.IntegerField(), default=list)
    class_label_processed_id_dict = JSONField(default=dict)
    class_label_dict = JSONField(default=dict)
    upload = models.FileField(upload_to='archives/predefined_featurexml/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


    def get_class_label_processed_id_dict(self):
        return OrderedDict(sorted(self.class_label_processed_id_dict.items(), key=lambda t: t[0]))

    def __str__(self):
        return f"{self.name} - {self.description}"

    def delete(self, *args, **kwargs):
        full_media_path = os.path.join(settings.MEDIA_ROOT, self.upload.name)
        # don't remove setup media on autoclean - only remove the associated files
        if not "/media/setup/" in full_media_path:
            os.remove(full_media_path)
        super(GCMSPredefinedPeakDetectionFileSet, self).delete(*args, **kwargs)



# class PeakAlignmentResultModel(models.Model):
#     # holds a peak Alignment result
#     analysis_id = models.ForeignKey(MccImsAnalysisWrapper, on_delete=models.CASCADE)
#     file = models.FileField(upload_to='data/peak_alignment_result/')


def coerce_pdm_from_path(a_path):
    available_methods = list(MccImsAnalysis.AVAILABLE_PEAK_DETECTION_METHODS)
    available_methods.extend(MccImsAnalysis.AVAILABLE_EXTERNAL_PEAK_DETECTION_METHODS)
    available_method_names = [am.name for am in available_methods]

    stem = Path(a_path).stem
    stem = stem.upper()

    # just try - if any matches use that - otherwise custom
    for amn, am in zip(available_method_names, available_methods):
        if amn in stem:
            return am
    return ExternalPeakDetectionMethod.CUSTOM


def construct_custom_peak_detection_fileset_from_zip(zippath):
    """
    construct_unlinked_web_peak_detection_results_from_zip
    :param zippath:
    :return:
    """
    name = Path(zippath).stem + Path(zippath).suffix

    coerced_pdm = coerce_pdm_from_path(zippath)
    class_label_file, pdrs = MccImsAnalysis.read_in_custom_peak_detection(zippath, pdm=coerced_pdm)

    pdr_ids = []
    for pdr in pdrs:
        result_filename, pdm_name = UnlinkedWebPeakDetectionResult.fields_from_peak_detection_result(pdr)

        uwpdr = UnlinkedWebPeakDetectionResult(
            filename=result_filename,
            peak_detection_method_name=pdm_name,
            csv_file=ContentFile(pdr.export_as_csv(directory="", use_buffer=True), name=result_filename + ".csv")
        )
        uwpdr.save()
        pdr_ids.append(uwpdr.pk)

    custom_fileset = CustomPeakDetectionFileSet(name=name, class_label_processed_id_dict=class_label_file, peak_detection_results=pdr_ids)
    custom_fileset.save()
    return custom_fileset.pk


def construct_custom_feature_matrix_from_zip(zippath):
    """
    extract the class_label dict and feature matrix from the zip
    :param zippath:
    :return:
    """

    class_label_dict, feature_matrix = MccImsAnalysis.read_in_custom_feature_matrix(zip_path=zippath)
    # name = Path(zippath).stem + Path(zippath).suffix

    # coerce peak detection method
    coerced_pdm = coerce_pdm_from_path(zippath)

    pdm_name = coerced_pdm.name

    fm_fn = "{}_feature_matrix".format(pdm_name)
    buff = StringIO()
    feature_matrix.to_csv(buff, index=True, header=True, index_label="index")
    buff.seek(0)
    fm = FeatureMatrix(analysis=None,
                       name=fm_fn,
                       peak_detection_method_name=pdm_name,
                       file=ContentFile(buff.getvalue(), name=fm_fn + ".csv", ),
                       class_label_dict=class_label_dict, # possible for OrderedDict?
                       )
    fm.save()
    print(f"Created FeatureMatrix {fm.pk}")

    return fm.pk


def create_GCMS_peak_detection_fileset_from_zip(zippath):
    """
    construct_unlinked_web_peak_detection_results_from_zip
    :param zippath:
    :return:
    """
    name = Path(zippath).stem + Path(zippath).suffix

    # TODO coerced_pdm = coerce_pdm_from_path(zippath)   # ectend for gcms methods - eg in name zips
    coerced_pdm = GCMSPeakDetectionMethod.CUSTOM

    # class_label_file, pdrs = MccImsAnalysis.read_in_custom_peak_detection(zippath, pdm=coerced_pdm)

    # TODO improve description - parse description file / user input?
    gcms_pd_fileset = GCMSPeakDetectionFileSet(name=name, description=name, class_label_processed_id_dict=OrderedDict())

    # move to db - create fileField - then get path from that to create pdrs
    cpdr_ids, class_label_dict, class_label_processed_id_dict = create_gcms_pdr_from_zip(archive_path=zippath, pdm=coerced_pdm)

    gcms_pd_fileset.class_label_dict = class_label_dict
    gcms_pd_fileset.class_label_processed_id_dict = class_label_processed_id_dict
    gcms_pd_fileset.peak_detection_results = cpdr_ids
    gcms_pd_fileset.peak_detection_method_name = coerced_pdm
    gcms_pd_fileset.save()
    return gcms_pd_fileset.pk


def create_gcms_pdr_from_zip(archive_path, pdm):
    """
    Create GCMSUnlinkedPeakDetectionResults and class label dict from zip archive
    :param archive_path:
    :param pdm:
    :return:
    """
    # from breathpy.model.ProcessingMethods import GCMSPeakDetectionMethod
    from breathpy.model.BreathCore import MccImsAnalysis
    from breathpy.model.GCMSTools import filter_feature_xmls
    from django.core.files.base import ContentFile
    import zipfile
    from io import BytesIO

    result_list = []
    with zipfile.ZipFile(archive_path) as archive:  # "r" is default mode

        # is sorted
        candidate_lis = filter_feature_xmls(dir="", name_list=archive.namelist())

        for fn in candidate_lis:
            # unzip to upload dir via buffer to let django handle file colisions
            # import ipdb; ipdb.set_trace()
            # buffer = StringIO(archive.read(fn))

            # feature_xml_outname = f"{fn}{get_default_feature_xml_storage_suffix(suffix_prefix=GCMSPeakDetectionMethod.ISOTOPEWAVELET.name)}"
            # feature_xml_name = f"{fn}{get_default_feature_xml_storage_suffix()}"
            # TODO potential to also guess the peakDetectionMethod from the filename
            buffer = BytesIO(archive.read(fn))
            gcms_updr = GCMSUnlinkedPeakDetectionResult(
                filename=fn,
                peak_detection_method_name=pdm.name,
                file=ContentFile(buffer.getvalue(), name=fn)
            )
            gcms_updr.save()

            result_list.append(gcms_updr.pk)
            # archive.close(fn)
        class_label_file_name = MccImsAnalysis.guess_class_label_extension(dir_to_search=archive_path,
                                                                           file_list_alternative=archive.namelist())
        label_dict = MccImsAnalysis.parse_class_labels_from_ZipFile(archive, class_label_file_name)
    class_label_processed_id_dict = {gcpdr_id: class_label for gcpdr_id, (fn, class_label) in zip(result_list, label_dict.items())}

    return result_list, label_dict, class_label_processed_id_dict


def create_gcms_raw_files_from_zip(zip_path, class_label_dict):
    """
    Create raw GCMS files for usage in GCMS fileset - basically just unzipping to a target directory
    :param zip_path:
    :param class_label_dict:
    :return:
    """
    raw_file_ids = []
    with ZipFile(zip_path) as archive:
        raw_gcms_fns = filter_mzml_or_mzxml_filenames(dir="", filelis=archive.namelist())

        for fn in raw_gcms_fns:
            # unzip to upload dir via buffer to let django handle file collisions

            buffer = BytesIO(archive.read(fn))
            raw_file = GCMSRawMeasurement(
                name=fn,
                label=class_label_dict[fn],
                file=ContentFile(buffer.getvalue(), name=fn)
            )
            raw_file.save()

            raw_file_ids.append(raw_file.pk)
    return raw_file_ids


def create_gcms_fileset_from_zip(zip_file_path, use_raw_files=True):
    """
    Create a basic `GCMSfileSet` from a zip - will extract rawfiles unless flagged otherwise
    :param zip_path:
    :param use_raw_files:
    :return: pk of `GCMSFileSet`
    """
    set_name = str(Path(zip_file_path).stem)
    with ZipFile(zip_file_path) as archive:
        class_label_file_name = MccImsAnalysis.guess_class_label_extension(dir_to_search=zip_file_path,
                                                                           file_list_alternative=archive.namelist())
        label_dict = MccImsAnalysis.parse_class_labels_from_ZipFile(archive, class_label_file_name)

    if use_raw_files:
        raw_file_ids = create_gcms_raw_files_from_zip(zip_path=zip_file_path, class_label_dict=label_dict)
        gcms_fileset = GCMSFileSet(name=set_name, raw_files=raw_file_ids, class_label_processed_id_dict=label_dict)

    else:
        # create with featureXML files
        gcms_PDFS_id = create_GCMS_peak_detection_fileset_from_zip(zip_file_path)
        gcms_pdfs = GCMSPeakDetectionFileSet.objects.get(pk=gcms_PDFS_id)

        # TODO actually guess the method from zip name?
        guessed_peak_detection_method_name = GCMSPeakDetectionMethod.CUSTOM.name
        # create fileset using peak detection fileset object
        gcms_fileset = GCMSFileSet(name=set_name, peak_detection_fileset=gcms_pdfs,
                              class_label_processed_id_dict=gcms_pdfs.class_label_dict,)

    gcms_fileset.save()
    return gcms_fileset.pk