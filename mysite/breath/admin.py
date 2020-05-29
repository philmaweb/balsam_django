from django.contrib import admin

# Register your models here.

from .models import (WebImsSet,
                     MccImsAnalysisWrapper,
                     RawFile,
                     ProcessedFile,
                     PredefinedFileset,
                     PredefinedCustomPeakDetectionFileSet,
                     FileSet,
                     ClassPredictionFileSet,
                     IntensityPlotModel,
                     OverlayPlotModel,
                     ClasswiseHeatMapPlotModel,
                     FeatureMatrix,
                     PredictionResult,
                     WebPredictionModel,
                     WebPeakDetectionResult,
                     DecisionTreePlotModel,
                     RocPlotModel,
                     GCMSUnlinkedPeakDetectionResult,
                     GCMSPredefinedPeakDetectionFileSet,
                     GCMSRawMeasurement,
                     GCMSPeakDetectionResult,
                     GCMSPeakDetectionFileSet,
                     )

admin.site.register(WebImsSet)
admin.site.register(MccImsAnalysisWrapper)
admin.site.register(RawFile)
admin.site.register(ProcessedFile)
admin.site.register(FileSet)
admin.site.register(PredefinedFileset)
admin.site.register(PredefinedCustomPeakDetectionFileSet)
admin.site.register(IntensityPlotModel)
admin.site.register(OverlayPlotModel)
admin.site.register(ClasswiseHeatMapPlotModel)
admin.site.register(FeatureMatrix)
admin.site.register(WebPredictionModel)
admin.site.register(WebPeakDetectionResult)
admin.site.register(PredictionResult)
admin.site.register(ClassPredictionFileSet)
admin.site.register(DecisionTreePlotModel)
admin.site.register(RocPlotModel)
admin.site.register(GCMSUnlinkedPeakDetectionResult)
admin.site.register(GCMSPredefinedPeakDetectionFileSet)
admin.site.register(GCMSRawMeasurement)
admin.site.register(GCMSPeakDetectionResult)
admin.site.register(GCMSPeakDetectionFileSet)
