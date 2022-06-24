import json

from django.shortcuts import render
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view
from django.http import HttpResponse
from django.http import JsonResponse


import os
import glob
import requests

from model.forms import AudioForm
from model.predict import load_model, predict
from model.preproc import load_audio, get_feature, convert_mp3_to_wav


@api_view(["POST"])
def Audio_store(request):
    form = AudioForm(request.POST, request.FILES or None)
    if form.is_valid():
        form.save()
    #   convert_mp3_to_wav("E:\paper_work\project\ser\media\documents\\"+request.FILES.get('record').name)
        audio = load_audio("E:\paper_work\project\ser\media\documents\\"+request.FILES.get('record').name)
        audio_date = get_feature(audio)
        model = load_model()
        output = predict(model, audio_date)
    #    return HttpResponse(output)
        return JsonResponse({'emotion': output})
    return JsonResponse({'emotion': 'bad'})