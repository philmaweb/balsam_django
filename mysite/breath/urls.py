from django.urls import path
from django.conf.urls import include, url
from django.conf.urls.static import static
from django.conf import settings

from django.contrib.auth.decorators import login_required

from . import views, views_gcms, views_cookie

urlpatterns = [
    # ex: /breath/
    path('', views.index, name='index'),

    # cookie consent urls
    # path('cookie_consent', views_cookie.TestPageView.as_view(), name='cookie_consent'),
    # path('', include('breath.urls_cookie')),

    # USER MANAGEMENT
    #Add Django site authentication urls (for login, logout, password management)
    path('accounts/', include('django.contrib.auth.urls')),

    # registration / register / signup
    path('signup/', views.signup, name="signup"),
    # TODO email confirmation, token, one time link

    path('logout/', views.lazy_logout, name="lazy_logout"),

    # ex: /breath/user/5
    # path('user/<int:user_id>', views.user, name='user'),

    # DATA PROCESSING

    # view available datasets
    path('list_datasets', views.list_datasets, name='list_datasets'),

    # upload new dataset and split it according to form input
    path('upload_dataset', views.upload_dataset, name='upload_dataset'),

    # delete user defined fileset
    path('delete_user_fileset', views.delete_user_fileset, name='delete_user_fileset'),

    # delete user defined feature matrix
    path('delete_user_fileset', views.delete_user_feature_matrix, name='delete_user_fileset'),

    # ex: /run/ # select between mccims and gcms
    path('run', views.run, name='run'),

    # ex: /run_mcc/ # select between automatic, custom and existing
    path('run_mcc', views.run_mcc, name='run_mcc'),

    # MCC URLS
    # ex: /selectdatasetauto/ --> reviewauto
    path('selectdatasetauto', views.selectDatasetAuto, name='selectDatasetAuto'),

    # ex: /reviewauto/  --> progressbar --> prediction
    path('reviewauto', views.reviewAuto, name='reviewAuto'),

    # ex: /selectdataset/
    path('selectdataset', views.selectDataset, name='selectDataset'),
    # ex: /selectparameters/
    path('selectparameters', views.selectParameters, name='selectParameters'),

    # path matching the exposed urls by djcelery for status checking
    path('task/<str:task_id>/status/', views.task_status, name='task_status'),

    # ex: /analysis/123
    path('analysis/<int:analysis_id>', views.analysis, name='analysis'),

    # ex: /analysis_progress/uuid
    path('analysis_progress/<str:task_id>/<int:analysis_id>', views.analysis_progress, name='analysis_progress'),

    # ex: /automatic_analysis_progress/uuid
    path('automatic_analysis_progress/<str:task_id>/<int:analysis_id>', views.automatic_analysis_progress, name='automatic_analysis_progress'),

    # ex: /evaluate_performance_progress/uuid
    path('evaluate_performance_progress/<str:task_id>/<int:analysis_id>', views.evaluate_performance_progress,
         name='evaluate_performance_progress'),

    # ex: /prediction_progress/uuid
    path('prediction_progress/<str:task_id>/<int:analysis_id>', views.prediction_progress, name='prediction_progress'),

    # ex: /prediction/17
    path('prediction/<int:analysis_id>', views.prediction, name='prediction'),

    # ex: /prediction_result/42
    path('prediction_result/<int:analysis_id>', views.prediction_result, name='prediction_result'),

    # ex: /analysis_result/
    # ex: /review/
    path('review', views.review, name='review'),


    # GENERAL urls
    # ex: /results/
    path('results', views.analysis_list, name='results'),

    # download for predictor pickle
    # path('predictor_pickle_download/<int:analysis_id>', views.predictor_pickle_download, name='predictor_pickle_download'),

    # ex: /about/
    path('about', views.about, name='about'),

    # ex: /documentation/
    path('documentation', views.documentation, name='documentation'),

    # ex: /help/
    path('help', views.help, name='help'),

    # ex: /analysis_details/3
    path('analysis_details/<int:analysis_id>', views.analysis_details, name='analysis_details'),

    # ex: /custom_detection_analysis
    path('custom_detection_analysis', views.custom_detection_analysis, name='custom_detection_analysis'),

    # ex: /custom_evaluation_params
    path('custom_evaluation_params/<int:analysis_id>/<int:is_using_fm>', views.custom_evaluation_params, name='custom_evaluation_params'),

    # ex: /custom_evaluation_progress/uuid
    path('custom_evaluation_progress/<str:task_id>/<int:analysis_id>', views.custom_evaluation_progress, name='custom_evaluation_progress'),

    # ex: /custom_prediction_result/42
    path('custom_prediction/<int:analysis_id>', views.custom_prediction, name='custom_prediction'),

    # ex: /custom_prediction_progress/uuid
    path('custom_prediction_progress/<str:task_id>/<int:analysis_id>', views.custom_prediction_progress, name='custom_prediction_progress'),

    # ex: /custom_prediction_result/42
    path('custom_prediction_result/<int:analysis_id>', views.custom_prediction_result, name='custom_prediction_result'),

    ###########
    # GCMS urls
    # ex: /selectdataset_gcms/ # select between raw and existing / featureXML
    path('select_dataset_gcms', views_gcms.selectDatasetGCMS, name='select_dataset_gcms'),

    # ex: selectparameters_gcms/
    path('select_parameters_gcms/<int:analysis_id>', views_gcms.selectParametersGCMS, name='select_parameters_gcms'),

    path('gcms_evaluation/<int:analysis_id>', views_gcms.gcms_evaluation, name='gcms_evaluation'),

    path('gcms_prediction/<int:analysis_id>', views_gcms.gcms_prediction, name='gcms_prediction'),

    path('gcms_prediction_result/<int:analysis_id>', views_gcms.gcms_prediction_result, name='gcms_prediction_result'),

    path('gcms_preprocessing_progress/<str:task_id>/<int:analysis_id>', views_gcms.gcms_preprocessing_progress, name='gcms_preprocessing_progress'),

    path('gcms_evaluation_progress/<str:task_id>/<int:analysis_id>', views_gcms.gcms_evaluation_progress, name='gcms_evaluation_progress'),

    path('gcms_prediction_progress/<str:task_id>/<int:analysis_id>', views_gcms.gcms_prediction_progress, name='gcms_prediction_progress'),

    ###########
    # Export urls
    path('get_trainings_matrix_as_json/<int:fm_id>', views.get_trainings_matrix_as_json, name='get_trainings_matrix_as_json'),
    path('get_trainings_matrix_as_csv/<int:fm_id>', views.get_trainings_matrix_as_csv, name='get_trainings_matrix_as_csv'),

    path('download_user_feature_matrix_csv/<int:fm_id>', views.download_user_feature_matrix_csv, name='download_user_feature_matrix_csv'),
    path('download_user_fileset_zip/<int:fm_id>', views.download_user_fileset_zip, name='download_user_fileset_zip'),
    path('download_default_fileset_zip/<int:fm_id>', views.download_default_fileset_zip, name='download_default_fileset_zip'),

    path('get_plots_as_archive/<int:analysis_id>', views.get_plot_archive, name='get_plots_as_archive'),
]



# only use when using Debug mode on the development server
if settings.DEBUG:
    # to serve stuff from media
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    # /test
    #     # import debug_toolbar
    #     # urlpatterns = [
    #     #       url(r'^__debug__/', include(debug_toolbar.urls)),
    #     #               ] + urlpatterns

else:
    # media url for authenticated view -> filter if user has access and let nginx serve
    urlpatterns += url(r'^media/(?P<path>.*)', views.media_access, name="media"),