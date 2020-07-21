import os
import time
import sys

import psycopg2
from django.contrib.auth.models import User
from .models import PredefinedFileset, create_gcms_pdr_from_zip, AnalysisType


def create_dbs():
    '''
    Requires environment variable 'DJANGO_SETTINGS_MODULE'
    Create databases from django settings, supports postgresql
    '''
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            print("create_dbs: let's go.")
            django_settings = __import__(os.environ['DJANGO_SETTINGS_MODULE'])#, fromlist='DATABASES')
            print("create_dbs: got settings.")
            databases = django_settings.DATABASES
            for name, db in databases.iteritems():
                host = db['HOST']
                user = db['USER']
                password = db['PASSWORD']
                port = db['PORT']
                db_name = db['NAME']
                db_type = db['ENGINE']
                # see if it is postgres
                if db_type.endswith('postgresql_psycopg2'):
                    print('creating database %s on %s' % (db_name, host))
                    con = psycopg2.connect(host=host, user=user, password=password, port=port, database='postgres')
                    con.set_isolation_level(0)
                    cur = con.cursor()
                    try:
                        cur.execute('CREATE DATABASE %s' % db_name)
                    except psycopg2.ProgrammingError as detail:
                        print(detail)
                        print('moving right along...')
                    exit(0)
                else:
                    print("ERROR: {0} is not supported by this script, you will need to create your database by hand.".format(db_type))
                    exit(1)
        except psycopg2.OperationalError:
            print("Could not connect to database. Waiting a little bit.")
            time.sleep(10)

    print('Could not connect to database after 1 minutes. Something is wrong.')
    exit(1)

def add_predefined_raw_filesets():
    import django
    from django.conf import settings

    from breath import models
    print("Populating DB with sample Datasets")
    names = [
             'Candy Train', 'Candy Validate',
             ]
    descriptions = [
                    'citrus vs menthol, 16 vs 17',
                    'citrus vs menthol, 4 vs 5',
                    ]
    initial_label = {}
    zip_folder = 'setup/'
    archive_names = [
                     'train_full_candy.zip', 'test_full_candy.zip',
                     ]
    archive_paths = [f'{zip_folder}{archive_name}' for archive_name in archive_names]


    train_val_ratios = [0.8, 0.2]
    is_trains = [True, False]

    for i, (name, description, archive_path, train_val_ratio, is_train) in enumerate(zip(names, descriptions, archive_paths, train_val_ratios, is_trains)):
        predefined_fileset = PredefinedFileset(name=name, description=description, upload=archive_path,
                                               filename_class_label_dict=initial_label,
                                               train_val_ratio=train_val_ratio, is_train=is_train,
                                               analysis_type=AnalysisType.RAW_MCC_IMS.name,
                                               )
        predefined_fileset.save()
        print(f"[{i}/{len(names) - 1}] Added {name}")

def add_predefined_peak_detection_filesets():
    from .models import PredefinedCustomPeakDetectionFileSet
    print("Populating DB with sample Peak Detection Results")
    # names need to be unique - otherwise key conflict
    base_names = [
             'Candy Train', 'Candy Validate',
             'Mouthwash Train', 'Mouthwash Validate',
             ]
    base_descriptions = [
                    'Citrus vs Menthol, 16 vs 17',
                    'Citrus vs Menthol, 4 vs 5',
                    '7 Mouthwashes, 6 samples each',
                    '7 Mouthwashes, 1 sample each',
                    ]
    initial_label = {}
    zip_folder = 'setup/'
    base_archive_names = [
                     'train_full_candy.zip', 'test_full_candy.zip',
                     'train_mouthwash.zip', 'test_mouthwash.zip',
                     ]

    base_split_fractions = [
            0.8, 0.2,
            0.85, 0.15
            ]
    base_analysis_types = ["PREPROCESSED_MCC_IMS", ]*4
    base_is_trains = [True, False, ] * 2

    # append results suffix and pdm name
    from breathpy.model.ProcessingMethods import PeakDetectionMethod, ExternalPeakDetectionMethod
    pdms_to_include = [ExternalPeakDetectionMethod.PEAX, PeakDetectionMethod.TOPHAT, PeakDetectionMethod.VISUALNOWLAYER]

    descriptions, names, archive_names, pdms = [], [], [], []
    split_fractions, analysis_types, is_trains = [], [], []

    for i, (bn, bd, ban) in enumerate(zip(base_names, base_descriptions, base_archive_names)):
        for pdm in pdms_to_include:

            pdm_name = pdm.name

            description_l = f"{bd} {pdm_name} "
            name_l = bn
            archive_nl = f"{ban[:-4]}_{pdm_name}_results.zip"

            descriptions.append(description_l)
            names.append(name_l)
            archive_names.append(archive_nl)
            pdms.append(pdm)

            split_fractions.append(base_split_fractions[i])
            analysis_types.append(base_analysis_types[i])
            is_trains.append(base_is_trains[i])

    archive_paths = [f'{zip_folder}{archive_name}' for archive_name in archive_names]

    for i, (name, description, archive_path, pdm, split_fraction, analysis_type, is_train) in enumerate(zip(names, descriptions, archive_paths, pdms, split_fractions, analysis_types, is_trains)):
        # read in pdr - and create model instances
        predefined_fileset = PredefinedCustomPeakDetectionFileSet(
                    name=name, description=description, upload=archive_path, class_label_processed_id_dict=initial_label,
                    train_val_ratio=split_fraction, analysis_type=analysis_type, is_train=is_train,
        )
        predefined_fileset.save()

        # move to db - create fileField - then get path from that to create pdrs

        cpdr_ids = create_cpdr_from_zip(archive_path=predefined_fileset.upload.path, pdm=pdm)
        predefined_fileset.peak_detection_results = cpdr_ids
        predefined_fileset.save()
        # except
        print(f"[{i}/{len(names) - 1}] Added {name}")


def create_cpdr_from_zip(archive_path, pdm):
    # from pathlib import Path
    from breathpy.model.BreathCore import MccImsAnalysis
    from .models import UnlinkedWebPeakDetectionResult
    from django.core.files.base import ContentFile

    class_label_file, pdrs = MccImsAnalysis.read_in_custom_peak_detection(archive_path, pdm=pdm)
    # name = Path(archive_path).stem + Path(archive_path).suffix

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
    return pdr_ids



def add_predefined_gcms_filesets():
    from .models import GCMSPredefinedPeakDetectionFileSet, GCMSPeakDetectionResult
    print("Populating DB with sample GCMS Peak Detection Results")
    # names need to be unique - otherwise key conflict
    base_names = [
             'Algae Train', 'Algae Validate',
             ]
    base_descriptions = [
                    '4 classes, 4 light, 3 dark, 4 n-limited and 4 replete.',
                    '4 classes, 1 light, 1 dark, 1 n-limited and 1 replete.',
                    ]
    zip_folder = 'setup/'
    base_archive_names = [
                     'train_algae.zip', 'test_algae.zip',
                     ]
    # train val ratio
    # 4/15 = 0.266
    from breathpy.model.ProcessingMethods import GCMSPeakDetectionMethod
    pdms_to_include = [GCMSPeakDetectionMethod.ISOTOPEWAVELET]
    sets_to_avoid = []
    descriptions, names, archive_names, pdms = [], [], [], []

    for bn, bd, ban in zip(base_names, base_descriptions, base_archive_names):
        for pdm in pdms_to_include:

            use = 1
            if bn in sets_to_avoid:
                if pdm in sets_to_avoid[bn]:
                    use = 0

            if use:
                pdm_name = pdm.name

                description_l = f"{bd} {pdm_name} "
                name_l = bn
                # archive_nl = f"{ban[:-4]}_{pdm_name}_results.zip"
                # don't have PDM in zip name for algae fileset
                archive_nl = f"{ban[:-4]}.zip"

                descriptions.append(description_l)
                names.append(name_l)
                archive_names.append(archive_nl)
                pdms.append(pdm)

    archive_paths = [f'{zip_folder}{archive_name}' for archive_name in archive_names]

    initial_label = {}
    for i, (name, description, archive_path, pdm) in enumerate(zip(names, descriptions, archive_paths, pdms)):
        # read in pdr - and create model instances

        # PredefinedCustomPeakDetectionFileSet.objects
        # try:
        is_train = (i % 2 == 0)
        if is_train:
            train_ratio = 0.8
        else:
            train_ratio = 0.2


        predefined_fileset = GCMSPredefinedPeakDetectionFileSet(name=name, description=description, upload=archive_path,
                                                                class_label_processed_id_dict=initial_label,
                                                                is_train=is_train, analysis_type="FEATURE_XML",
                                                                train_val_ratio=train_ratio)
        predefined_fileset.save()

        # move to db - create fileField - then get path from that to create pdrs
        cpdr_ids, class_label_dict, class_label_processed_id_dict = create_gcms_pdr_from_zip(archive_path=predefined_fileset.upload.path, pdm=pdm)

        predefined_fileset.class_label_dict = class_label_dict
        predefined_fileset.class_label_processed_id_dict = class_label_processed_id_dict
        predefined_fileset.peak_detection_results = cpdr_ids
        predefined_fileset.save()
        # except
        print(f"[{i}/{len(names) - 1}] Added {name}")


def populate_db():
    add_predefined_raw_filesets()
    add_predefined_peak_detection_filesets()
    add_predefined_gcms_filesets()

def create_cookies():
    """
    Create cookies for cookie consent
    :return:
    """
    from cookie_consent.models import CookieGroup, Cookie
    cookie_group = CookieGroup(varname="mandatory",
                               description="Mandatory Cookies",
                               is_required=True,
                               is_deletable=True)
    cookie_group.save()
    # cookie_group = CookieGroup.objects.get(pk=1)
    csrftoken = Cookie(cookiegroup=cookie_group, name="csrftoken", domain="exbio.wzw.tum.de/balsam")
    csrftoken.save()
    sessionid = Cookie(cookiegroup=cookie_group, name="sessionid", domain="exbio.wzw.tum.de/balsam")
    sessionid.save()
    print(f"Created {csrftoken} and {sessionid}")


# how to reset and recreate db:
# flush does not delete the tables - only data within
# python manage.py flush
# python manage.py makemigrations
# python manage.py migrate
# python manage.py createsuperuser


if __name__ == '__main__':
    print("create_dbs start")
    populate_db()
    create_cookies()
    print("create_dbs all done")