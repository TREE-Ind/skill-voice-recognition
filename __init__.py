# Copyright 2016 Mycroft AI, Inc.
#
# This file is part of Mycroft Core.
#
# Mycroft Core is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mycroft Core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mycroft Core.  If not, see <http://www.gnu.org/licenses/>.

from adapt.intent import IntentBuilder

from mycroft.skills.core import MycroftSkill
from mycroft.util.log import getLogger

import os
import sys
import re
import fileinput
import tflearn


sys.path.append('/opt/mycroft/skills/skill-voice-recognition')
import speech_data as data

import tensorflow as tf
tf.reset_default_graph()

__author__ = 'TREE'

LOGGER = getLogger(__name__)


class SpeakerRecognitionSkill(MycroftSkill):
    def __init__(self):
        super(SpeakerRecognitionSkill, self).__init__(name="SpeakerRecognitionSkill")


    def initialize(self):

        speaker_rec_test_intent = IntentBuilder("SpeakerRecTestIntent"). \
            require("SpeakerRecTestKeyword").build()
        self.register_intent(speaker_rec_test_intent, self.handle_speaker_rec_test_intent)

        start_voice_training_intent = IntentBuilder("StartVoiceTrainingIntent"). \
            require("StartVoiceTrainingKeyword").require("User").build()
        self.register_intent(start_voice_training_intent, self.handle_start_voice_training_intent)

        end_voice_training_intent = IntentBuilder("EndTrainingIntent"). \
            require("EndVoiceTrainingKeyword").build()
        self.register_intent(end_voice_training_intent, self.handle_end_voice_training_intent)

    def handle_start_voice_training_intent(self, message):
        #TODO this is where we will start the voice training process
        user = message.data.get("User")
        text = "record_wake_words"   # Search for config variable to change.
        new_text = "    \"record_wake_words\": true,\n"
        x = fileinput.input(files="~/mycroft-core/mycroft/configuration/mycroft.conf", inplace=1)
        for line in x:
            if text in line:
                line = new_text
            print line,
        x.close()
        self.speak("Voice training has been enabled for %s" % (user))
        self.speak("Use skills normally for a short while then say, end voice training")

    def handle_end_voice_training_intent(self, message):
        #TODO this is where we will end the voice training process
        text = "record_wake_words"   # if any line contains this text, I want to modify the whole line.
        new_text = "    \"record_wake_words\": false,\n"
        x = fileinput.input(files="~/mycroft-core/mycroft/configuration/mycroft.conf", inplace=1)
        for line in x:
            if text in line:
                line = new_text
            print line,
        x.close()
        self.speak("Voice training complete, I should now be able to recognize your voice.  \
            If you feel like I'm not being accurate enough, please enable voice training again")

    def handle_speaker_rec_test_intent(self, message):
        speakers = data.get_speakers()
        number_classes=len(speakers)
        #print("speakers",speakers)

        #batch=data.wave_batch_generator(batch_size=1000, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
        #X,Y=next(batch)


        # Classification
        #tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

        net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
        net = tflearn.fully_connected(net, 64)
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, number_classes, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

        model = tflearn.DNN(net)
        #model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)

        CWD_PATH = os.path.dirname(__file__)
        path_to_model = os.path.join(CWD_PATH, 'model', 'model.tfl')
        model.load(path_to_model) 

        demo_file = "8_Vicki_260.wav"
        #demo_file = "8_Bruce_260.wav"
        demo=data.load_wav_file(data.path + demo_file)
        result=model.predict([demo])
        result=data.one_hot_to_item(result,speakers)
        if result == "Vicki":
            self.speak("I am confident I'm speaking to %s"%(result)) # ~ 97% correct
        else:
            self.speak("I'm sorry I don't recognize your voice")


    def stop(self):
        pass


def create_skill():
    return SpeakerRecognitionSkill()
