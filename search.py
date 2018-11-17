#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import list_file as lf
import speech_recognition as sr
import pyttsx3

# obtain audio from the microphone
def Search():
    print('HEllo')
    r = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        print("How can I help you?\n")
        audio = r.listen(source)

    # recognize speech using Houndify
    HOUNDIFY_CLIENT_ID = "uxIWu9AHQuIwTgiyqZ47Gw=="  # Houndify client IDs are Base64-encoded strings
    HOUNDIFY_CLIENT_KEY = "tXE7v6eFSqb_ZfRJ8jaXfntsMbrNgRMHrh3Re2Lv5DbhqBDy3z0ax4ZkylbfnJqzPBeC_jsf1NOYavPJ26DSKw=="  # Houndify client keys are Base64-encoded strings

    TRIGGER = True
    while (TRIGGER):
        try:
            message = r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
            print(message+'\n')
            TRIGGER = False
        except sr.UnknownValueError:
            print("Houndify could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Houndify service; {0}".format(e))

    pos_info = {}
    pos_info = lf.getdic('data.txt')
    # print(pos_info)
    engine = pyttsx3.init()

    for k in pos_info.keys():
        if k in message:
            voice = "It is on the {}".format(pos_info[k])
            print(voice)
            engine.say(voice)
            engine.runAndWait()

if __name__ == '__main__':
    Search()
