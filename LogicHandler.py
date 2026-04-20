import base64
import asyncio
import sys
from fastapi import WebSocket
import tempfile
import os

from MainServerHelper.Pydantic_frame import DiscordBotTextRequest, ReminhInferenceRequest, STTRequest, TTSRequest, MainAiRequest, ReloadYamlRequest
from MainServerHelper.Pydantic_frame import DiscordBotTextResponse
from VL.VisionLangHandler import VisionLangHandler
from TTS.TTS_Handler import TTS_Handler
from STT.STT_handler import STT_handler
# Reminh
from Persona.Reminh import Reminh

class ReminhLogicHandler:
    def __init__(self, vl_handler: VisionLangHandler, tts_handler: TTS_Handler, stt_handler:STT_handler):
        self.vl_handler: VisionLangHandler = vl_handler
        self.tts_handler: TTS_Handler = tts_handler
        self.stt_handler: STT_handler = stt_handler

    async def handle_main_inference(self, Remi:Reminh, websocket: WebSocket, request: ReminhInferenceRequest):
        print(f"[LogicHandler] Main Inference Start: {request.input_type}")
        try:
            current_prompt = request.text or ""
            
            # 1. STT
            if "audio" in request.input_type and request.audio_bytes:
                print("[LogicHandler] Audio input detected - Decoding Base64")
    
                try:
                    audio_data = request.audio_bytes
                    if "," in audio_data:
                        audio_data = audio_data.split(",")[1]
            
                    decoded_audio = base64.b64decode(audio_data)

                    with tempfile.NamedTemporaryFile(mode='wb', suffix=".webm", delete=False) as temp_audio:
                        temp_audio.write(decoded_audio)
                    temp_path = temp_audio.name

                    try:
                        stt_res = self.stt_handler.set_data(temp_path)
                        if stt_res and stt_res.get('text'):
                            transcribed_text = stt_res['text']
                        current_prompt = f"{current_prompt} {transcribed_text}".strip()
            
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                except Exception as e:
                    print(f"[STT Error] Failed to decode audio: {e}") 

            # 2-1. VAD_inference
            system_prompt = Remi.get_VAD_prompt() 
            
            vlm_response_container = self.vl_handler.inference(
                user_prompt=(f"User input: {current_prompt}"),
                system_prompt=system_prompt,
                image_path=request.image_base64
            )

            vlm_response_text = ""
            if vlm_response_container and vlm_response_container[0]:
                print("[LogicHandler] VLM inference done! (VAD)")
                print(f"DEBUG VLM RESULT: {vlm_response_container}")
                vlm_response_text = vlm_response_container[1]
                print(vlm_response_text)
            else:
                print("[LogicHandler] Warning: No text generated from VLM. (VAD)")
                await websocket.send_json({"error": "VLM Generation Failed (VAD)"})
                return

            # 2-2. Reminh inference
            system_prompt = Remi.get_Reminh_prompt(vlm_response_text, current_prompt)

            current_captured_emotion = Remi.get_Reminh_last_emotion().copy()

            vlm_response_container = self.vl_handler.inference(
                    user_prompt=current_prompt,
                    system_prompt=system_prompt,
                    image_path=request.image_base64
                    )

            if vlm_response_container and vlm_response_container[0]:
                print("[LogicHandler] VLM inference done! (Reminh)")
                print(f"DEBUG VLM RESULT: {vlm_response_container}")
                vlm_response_text = vlm_response_container[1]
                print(vlm_response_text)
            else:
                print("[LogicHandler] Warning: No text generated from VLM. (Reminh)")
                await websocket.send_json({"error": "VLM Generation Failed (Reminh)"})
                return

            # 3. TTS Inference
            audio_b64_str = None
            if vlm_response_text:
                # TODO: Path goes to config
                audio_raw_bytes = self.tts_handler.speak(
                    character_dialog=vlm_response_text,
                    ref_audio_path="/home/Reminh/work/deltaAnima/Reminh/TTS/default_ref.wav",
                    prompt_text="Hello people from Langara CTF team. My name is Reminh",
                    prompt_lang="en",
                    text_lang="en"
                )
                print("[LogicHandler] TTS Generation Done.")
                
                if audio_raw_bytes:
                    audio_b64_str = base64.b64encode(audio_raw_bytes).decode("utf-8")

            # 4. Send Response
            await websocket.send_json({
                "type": "audio_response",
                "text": vlm_response_text,
                "audio": audio_b64_str,
                "expre_result": current_captured_emotion
            })

        except Exception as e:
            print(f"[LogicHandler] Inference Error: {e}")
            await websocket.send_json({"error": f"Inference processing failed: {str(e)}"})

    async def handle_stt_test(self, websocket: WebSocket, request: STTRequest):
        # STT 
        pass

    async def handle_tts_test(self, websocket: WebSocket, request: TTSRequest):
        # TTS 
        pass

    async def handle_reload_yaml(self, Remi: Reminh, websocket: WebSocket, request: ReloadYamlRequest):
        print("[System] YAML reload request received")
    
        try:
            # Calling the direct binding function
            success: bool = Remi.reload_physics_config() 
        
            if success:
                print("[System] deltaEGO YAML update successful")
                await websocket.send_json({
                    "response_type": "system_notice",
                    "status": "success",
                    "message": "Reminh's emotion physics settings (YAML) have been successfully updated."
                })
            else:
                print("[System] deltaEGO YAML update failed")
                await websocket.send_json({
                    "response_type": "system_notice",
                    "status": "error",
                    "message": "Failed to update YAML. (Internal C++ Error)"
                })
            
        except Exception as e:
            print(f"[Fatal Error] Exception occurred during YAML reload: {e}")
            await websocket.send_json({
                "response_type": "system_notice",
                "status": "error",
                "message": str(e)
            })

    async def handle_discord_text_inference(self, Remi: Reminh, request: DiscordBotTextRequest):
        """
        [DISCORD ISOLATED INFERENCE]
        - No STT, No TTS (Pure Text/Vision focus)
        - Uses Discord-specific Prompting
        - USES HTTP
        """
        print(f"[LogicHandler] Discord isolated inference for: {request.user_name}")
        
        try:
            current_prompt = request.message_content

            target_image = request.attachments[0] if request.attachments else None

            system_prompt_vad = Remi.get_VAD_prompt()

            vlm_res_vad = self.vl_handler.inference(
                user_prompt=f"User: {current_prompt}",
                system_prompt=system_prompt_vad,
                image_path=target_image
            )

            vlm_vad_raw = vlm_res_vad[1] if vlm_res_vad and vlm_res_vad[0] else "{}"
            
            system_prompt_reminh = Remi.get_Reminh_prompt(
                VAD_result_raw=vlm_vad_raw,
                user_input=current_prompt,
                source="discord_txt",
                user_name=request.user_name
            )

            vlm_res_reminh = self.vl_handler.inference(
                user_prompt=current_prompt,
                system_prompt=system_prompt_reminh,
                image_path=target_image
            )

            Remi.set_Reminh_memory(User_Name=request.user_name, AI_output=vlm_res_reminh[1])

            last_emotion = Remi.get_Reminh_last_emotion()

            if vlm_res_reminh and vlm_res_reminh[0]:
                return DiscordBotTextResponse(
                    status = "success",
                    output_text = vlm_res_reminh[1],
                    emotion_tag = last_emotion.get("emotion_term", "calm"),
                )
            else:
                return DiscordBotTextResponse(status="error", output_text="VLM inference failed")

        except Exception as e:
            print(f"[Critical Discord Error] {e}")
            return DiscordBotTextResponse(status="error", output_text=str(e))
