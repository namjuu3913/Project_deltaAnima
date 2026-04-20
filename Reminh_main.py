"""
    external libs
"""
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from pydantic import ValidationError, TypeAdapter
from typing import Union
import json
import asyncio
import base64
import time

"""
    custom libs
"""
# pydantic classes
from MainServerHelper.Pydantic_frame import *
# setting up server
from setup_servers import *
from VL.VisionLangHandler import VisionLangHandler
from TTS.TTS_Handler import TTS_Handler
from STT.STT_handler import STT_handler as STT_Handler
# websocket logics
from LogicHandler import ReminhLogicHandler
# Reminh
from Persona.Reminh import Reminh

print("Welcome!!!")
print("Waking up Reminh...........")

"""
    # 1. setting up server
        * VL model server (5090)
        * STT server (3090)
        * TTS server (3090)
"""

print("Step 1. initializing isolated model servers..........")
setup_tmux()
print("servers are started!")

"""
   # 2. Initializing Handler
        * VisionLangHandler
        * TTS_Handler
"""
print("Step 2.Starting model handlers........")
VL_handler: VisionLangHandler = VisionLangHandler( 
        temperature = 0.8,
        max_token=3000,
        abs_model_path="~/work/deltaAnima/Reminh/models/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-Q4_K_M.gguf", 
        model_full="Qwen3-VL-30B-A3B-Instruct-abliterated"
        ) 
TTS_handler: TTS_Handler = TTS_Handler()
STT_handler: STT_Handler = STT_Handler(model="turbo")

logic_handler = ReminhLogicHandler(vl_handler=VL_handler, tts_handler=TTS_handler, stt_handler=STT_handler)
print("Loading Handlers are done!!!! Starting websocket server................")
# Waking up Reminh
Remi: Reminh = Reminh()

"""
    # 3. websocket server
"""
class ConnectionManager:
    def __init__(self):
        self.active_connection: Optional[WebSocket] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
        print("[Orchestrator] Unity Client Connected.")

    def disconnect(self):
        self.active_connection = None
        print("[Orchestrator] Client Disconnected.")

    async def send_json(self, data: dict):
        if self.active_connection:
            await self.active_connection.send_json(data)

manager = ConnectionManager()
request_adapter = TypeAdapter(
        Union[
            ReminhInferenceRequest,
            STTRequest,
            TTSRequest,
            MainAiRequest
            ]
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Starting Reminh Orchestrator...")
    yield
    print("[Server] Shutting down. Saving all memories...")
    try:
        Remi.MemoryHandler.save_db()
    except Exception as e:
        print(f"[Error] Failed to save memories on shutdown: {e}")
app = FastAPI(lifespan = lifespan)
@app.websocket("/ws/reminh")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_json()
            print(f"LOG: Raw_data: {raw_data}")
            
            # 1. Validate Request
            try:
                request = request_adapter.validate_python(raw_data)
            except ValidationError:
                await websocket.send_json({"error": "Unknown Request Type or Invalid Format"})
                continue
            
            if isinstance(request, ReminhInferenceRequest):
                await logic_handler.handle_main_inference(Remi, websocket, request)
            
            elif isinstance(request, STTRequest):
                await logic_handler.handle_stt_test(websocket, request)
                
            elif isinstance(request, TTSRequest):
                await logic_handler.handle_tts_test(websocket, request)
                
            elif isinstance(request, MainAiRequest):
                pass

            elif isinstance(request, ReloadYamlRequest):
                await logic_handler.handle_reload_yaml(Remi, websocket, request)

    except WebSocketDisconnect:
        print("[Reminh Orchestrator] Client disconnected")
        manager.disconnect()

    except Exception as e:
        print(f"[Fatal Error] {e}")

@app.post("/discord/chat")
async def discord_chat_http(request: DiscordBotTextRequest):
    response = await logic_handler.handle_discord_text_inference(Remi, request)
    return response.model_dump()

@app.post("/admin/reload")
async def reload_prompt():
    Remi.PromptModule.reload()
    return {"status": "success", "message": "Prompt reloaded"}
