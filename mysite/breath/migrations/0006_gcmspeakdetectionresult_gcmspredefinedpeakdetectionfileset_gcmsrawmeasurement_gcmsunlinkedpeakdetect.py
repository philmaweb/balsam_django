# Generated by Django 2.2 on 2019-11-11 17:57

import django.contrib.postgres.fields
import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('breath', '0005_fileset_created_at'),
    ]

    operations = [
        migrations.CreateModel(
            name='GCMSPredefinedPeakDetectionFileSet',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.CharField(max_length=100)),
                ('peak_detection_results', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=list, size=None)),
                ('class_label_processed_id_dict', django.contrib.postgres.fields.jsonb.JSONField(default=dict)),
                ('class_label_dict', django.contrib.postgres.fields.jsonb.JSONField(default=dict)),
                ('upload', models.FileField(upload_to='archives/predefined_featurexml/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='GCMSRawMeasurement',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('label', models.CharField(max_length=30)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('file', models.FileField(upload_to='data/raw_gcms/')),
            ],
        ),
        migrations.CreateModel(
            name='GCMSUnlinkedPeakDetectionResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=100)),
                ('file', models.FileField(upload_to='data/peak_detection_gcms/')),
                ('peak_detection_method_name', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='GCMSPeakDetectionResult',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.CharField(max_length=100)),
                ('file', models.FileField(upload_to='data/peak_detection_gcms/')),
                ('peak_detection_method_name', models.CharField(max_length=20)),
                ('raw_measurement', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='breath.GCMSRawMeasurement')),
            ],
        ),
    ]