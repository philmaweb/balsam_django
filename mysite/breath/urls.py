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

    # ex: /breath/run/ # select between mccims and gcms
    path('run', views.run, name='run'),

    # ex: /breath/run_mcc/ # select between automatic, custom and existing
    path('run_mcc', views.run_mcc, name='run_mcc'),

    # MCC URLS
    # ex: /breath/selectdatasetauto/ --> reviewauto
    path('selectdatasetauto', views.selectDatasetAuto, name='selectDatasetAuto'),

    # ex: /breath/reviewauto/  --> progressbar --> prediction
    path('reviewauto', views.reviewAuto, name='reviewAuto'),

    # ex: /breath/selectdataset/
    path('selectdataset', views.selectDataset, name='selectDataset'),
    # ex: /breath/selectparameters/
    path('selectparameters', views.selectParameters, name='selectParameters'),

    # path matching the exposed urls by djcelery for status checking
    path('task/<str:task_id>/status/', views.task_status, name='task_status'),

    # ex: /breath/analysis/123
    path('analysis/<int:analysis_id>', views.analysis, name='analysis'),

    # ex: /breath/analysis_progress/uuid
    path('analysis_progress/<str:task_id>/<int:analysis_id>', views.analysis_progress, name='analysis_progress'),

    # ex: /breath/automatic_analysis_progress/uuid
    path('automatic_analysis_progress/<str:task_id>/<int:analysis_id>', views.automatic_analysis_progress, name='automatic_analysis_progress'),

    # ex: /breath/evaluate_performance_progress/uuid
    path('evaluate_performance_progress/<str:task_id>/<int:analysis_id>', views.evaluate_performance_progress,
         name='evaluate_performance_progress'),

    # ex: /breath/prediction_progress/uuid
    path('prediction_progress/<str:task_id>/<int:analysis_id>', views.prediction_progress, name='prediction_progress'),

    # ex: /breath/prediction/17
    path('prediction/<int:analysis_id>', views.prediction, name='prediction'),

    # ex: /breath/prediction_result/42
    path('prediction_result/<int:analysis_id>', views.prediction_result, name='prediction_result'),

    # ex: /breath/analysis_result/
    # ex: /breath/review/
    path('review', views.review, name='review'),


    # GENERAL urls
    # ex: /breath/results/
    path('results', views.analysis_list, name='results'),

    # download for predictor pickle
    # path('predictor_pickle_download/<int:analysis_id>', views.predictor_pickle_download, name='predictor_pickle_download'),

    # ex: /breath/about/
    path('about', views.about, name='about'),

    # ex: /breath/documentation/
    path('documentation', views.documentation, name='documentation'),

    # ex: /breath/help/
    path('help', views.help, name='help'),

    # ex: /breath/analysis_details/3
    path('analysis_details/<int:analysis_id>', views.analysis_details, name='analysis_details'),


    # ex: /breath/custom_detection_analysis
    path('custom_detection_analysis', views.custom_detection_analysis, name='custom_detection_analysis'),

    # ex: /breath/custom_evaluation_params
    path('custom_evaluation_params/<int:analysis_id>/<int:is_using_fm>', views.custom_evaluation_params, name='custom_evaluation_params'),

    # ex: /breath/custom_evaluation_progress/uuid
    path('custom_evaluation_progress/<str:task_id>/<int:analysis_id>', views.custom_evaluation_progress, name='custom_evaluation_progress'),

    # ex: /breath/custom_prediction_result/42
    path('custom_prediction/<int:analysis_id>', views.custom_prediction, name='custom_prediction'),

    # ex: /breath/custom_prediction_progress/uuid
    path('custom_prediction_progress/<str:task_id>/<int:analysis_id>', views.custom_prediction_progress, name='custom_prediction_progress'),

    # ex: /breath/custom_prediction_result/42
    path('custom_prediction_result/<int:analysis_id>', views.custom_prediction_result, name='custom_prediction_result'),

    ###########
    # GCMS urls
    # ex: /breath/selectdataset_gcms/ # select between raw and existing / featureXML
    path('select_dataset_gcms', views_gcms.selectDatasetGCMS, name='select_dataset_gcms'),

    # ex: selectparameters_gcms/
    path('select_parameters_gcms/<int:analysis_id>', views_gcms.selectParametersGCMS, name='select_parameters_gcms'),

    path('gcms_evaluation/<int:analysis_id>', views_gcms.gcms_evaluation, name='gcms_evaluation'),

    path('gcms_prediction/<int:analysis_id>', views_gcms.gcms_prediction, name='gcms_prediction'),

    path('gcms_prediction_result/<int:analysis_id>', views_gcms.gcms_prediction_result, name='gcms_prediction_result'),

    path('gcms_preprocessing_progress/<str:task_id>/<int:analysis_id>', views_gcms.gcms_preprocessing_progress, name='gcms_preprocessing_progress'),

    path('gcms_evaluation_progress/<str:task_id>/<int:analysis_id>', views_gcms.gcms_evaluation_progress, name='gcms_evaluation_progress'),

    path('gcms_prediction_progress/<str:task_id>/<int:analysis_id>', views_gcms.gcms_prediction_progress, name='gcms_prediction_progress'),

    path('get_trainings_matrix_as_json/<int:fm_id>', views.get_trainings_matrix_as_json, name='get_trainings_matrix_as_json'),
    path('get_trainings_matrix_as_csv/<int:fm_id>', views.get_trainings_matrix_as_csv, name='get_trainings_matrix_as_csv'),

    path('get_plots_as_archive/<int:analysis_id>', views.get_plot_archive, name='get_plots_as_archive'),
]



# only use when using Debug mode on the development server
if settings.DEBUG:
    # to serve stuff from media
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    # /breath/test
    #     # import debug_toolbar
    #     # urlpatterns = [
    #     #       url(r'^__debug__/', include(debug_toolbar.urls)),
    #     #               ] + urlpatterns

else:
    # media url for authenticated view -> filter if user has access and let nginx serve
    urlpatterns += url(r'^media/(?P<path>.*)', views.media_access, name="media"),