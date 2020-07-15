from datetime import timedelta
from django.utils import timezone
from .models import WebImsSet, WebCustomSet, MccImsAnalysisWrapper, FileSet, UserDefinedFeatureMatrix, UserDefinedFileset


def clean_up(age_limit_days=30):
    """
    Get all relevant object instances and delete them
    :return: None
    """
    lis = []
    lis.extend(MccImsAnalysisWrapper.objects.filter(created_at__lte=timezone.now()-timedelta(days=age_limit_days)))
    lis.extend(WebImsSet.objects.filter(uploaded_at__lte=timezone.now()-timedelta(days=age_limit_days)))
    lis.extend(WebCustomSet.objects.filter(uploaded_at__lte=timezone.now() - timedelta(days=age_limit_days)))
    lis.extend(FileSet.objects.filter(created_at__lte=timezone.now() - timedelta(days=age_limit_days)))
    lis.extend(UserDefinedFeatureMatrix.objects.filter(created_at__lte=timezone.now() - timedelta(days=age_limit_days)))
    lis.extend(UserDefinedFileset.objects.filter(created_at__lte=timezone.now() - timedelta(days=age_limit_days)))

    for ro in lis:
        print(f"Deleting {ro}")
        try:
            ro.delete()
        except FileNotFoundError:
            # already removed, moving on
            pass
    if not len(lis):
        print("Nothing to clean up")


# if __name__ == '__main__':
#     print("cleanup start")
#     clean_up()
#     print("cleanup all done")