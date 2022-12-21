# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import re
import sys
import os
from google.cloud import speech
import pyaudio
import queue
from threading import Thread
import time
import pygame
# import playsound

#init
pygame.mixer.init()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"./changong-a2bfa0656046.json"

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.isPause = False

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()


        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def pause(self):
        if self.isPause == False:
            self.isPause = True


    def resume(self):
        if self.isPause == True:
            self.isPause = False


    def status(self):
        return self.isPause

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        if self.isPause == False:
            self._buff.put(in_data)
        #else
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return

            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


# [END audio_stream]



class Gspeech(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.language_code = 'ko-KR'  # a BCP-47 language tag
        self.cnt=0
        self._buff = queue.Queue()
        self.speech_text = []
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=self.language_code)
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True)

        self.mic = None
        self.status = True
        self.is_eaten = False
        self.daemon = True
        self.start()

    def __exit__(self):
        self._buff.put(None)

    def run(self):
        with MicrophoneStream(RATE, CHUNK) as stream:
            self.mic = stream
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = self.client.streaming_recognize(self.streaming_config, requests)

            # Now, put the transcription responses to use.
            self.listen_print_loop(responses, stream)
        self._buff.put(None)
        self.status = False

    def pauseMic(self):
        if self.mic is not None:
            self.mic.pause()

    def resumeMic(self):
        if self.mic is not None:
            self.mic.resume()

    # 인식된 Text 가져가기
    def getText(self, block = True):
        return self._buff.get(block=block)

    # 음성인식 처리 루틴
    def listen_print_loop(self, responses, mic):
        try:
            for response in responses:
                if not response.results:
                    # print('not')
                    continue

                result = response.results[0]
                if not result.alternatives:
                    # print('not')
                    continue
                
                # 3번의 되물음 다음 음성인식 시도 시 강제 종료
                if self.cnt >= 5:
                    break

                transcript = result.alternatives[0].transcript
                # overwrite_chars = ' ' * (num_chars_printed - len(transcript))
                # if not result.is_final:
                    # sys.stdout.write(transcript + overwrite_chars + '\r')
                    # sys.stdout.flush()
                    #### 추가 ### 화면에 인식 되는 동안 표시되는 부분.
                    # num_chars_printed = len(transcript)
                # else:
                    # self.speech_text.append(transcript)
                    # self._buff.put(transcript)
                    # num_chars_printed = 0

                if result.is_final:
                  # except the first input, transcription received by server .
                # include " " in first line. so remove it.
                    if transcript[0] == ' ':
                        transcription = transcript[1:]
                    else:
                        transcription = transcript
    
                    #if re.fullmatch('먹었어', transcription, re.I):
                    if '먹었어' in transcript :
                        time.sleep(1)
                        self.speech_text.append(transcription)
                        self.is_eaten = True
                        print(transcription, '먹었다고 말함')
                        # return 1
                        break
                    # print transcription.
                    # can't print out korean as pyautogui.typewrite, so use clipboard.
                    else:
                        time.sleep(1)
                        print(transcription)
                        # print(self.cnt)
                        if self.cnt >= 5:
                            print(self.cnt)
                        # return 2
                            break
        except:
            return

def main():
    # init_time = time.time()
    # gsp = Gspeech()
    # while True:
    #     print(1)
    #     # 음성 인식 될때까지 대기 한다.
    #     stt = gsp.getText()
    #     if stt is None:
    #         break
    #     print(stt)
    #     time.sleep(0.01)
    #     if ('끝내자' in stt):
    #         break
    gsp = Gspeech()
    # test=gsp.listen_print_loop()
    time.sleep(1)
    end_main = 0
   
    for i in range(5):
        if gsp.is_eaten:
            # print('hi')
            # playsound.playsound('./is_eaten.mp3')
            pygame.mixer.music.load("is_eaten.mp3")
            #play
            pygame.mixer.music.play()
            #끝까지 재생할때까지 기다린다.
            while pygame.mixer.music.get_busy() == True:
                continue
            end_main = 1
            return 1
        if i == 4:
            # playsound.playsound('./end_checking.mp3')
            pygame.mixer.music.load("end_checking.mp3")
            #play
            pygame.mixer.music.play()
            #끝까지 재생할때까지 기다린다.
            while pygame.mixer.music.get_busy() == True:
                continue
            end_main = 2
            return 2

        gsp.pauseMic()
        # time.sleep(1)
        #load file
        pygame.mixer.music.load("check_audio.mp3")
        #play
        pygame.mixer.music.play()
        #끝까지 재생할때까지 기다린다.
        while pygame.mixer.music.get_busy() == True:
            continue
        gsp.resumeMic()
        if end_main == 1:
            time.sleep(2)
            return 1
        elif end_main == 2:
            time.sleep(2)
            return 2 
        time.sleep(10)
        gsp.cnt+=1
        print('speech:',gsp.cnt)
        # print(gsp)

if __name__ == '__main__':
    main()
    # print(main())