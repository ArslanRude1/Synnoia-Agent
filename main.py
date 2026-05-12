import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
import asyncio

load_dotenv()

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from graph.graph import synnoia_agent
from json_converter.converter_tiptap_to_synnoia import tiptap_to_synnoia
from json_converter.converter_synnoia_to_tiptap import synnoia_to_tiptap

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "synnoia-agent"}


@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive data from frontend
            data = await websocket.receive_json()
            
            query = data.get("query", "")
            doc_text = data.get("doc_text", "")
            doc_json = data.get("doc_json", "")
            model = data.get("model", "gpt-5.4")
            
            print(f"Received request: query='{query}', doc_json_present={bool(doc_json)}")
            
            # Send acknowledgment
            await websocket.send_json({"status": "processing", "message": "Processing your request..."})
            
            # Prepare initial_state matching SynnoiaState schema
            initial_state = {
                "query": query,
                "doc_text": doc_text,
                "doc_json": doc_json,
                "model": model,
                "intent": "",
                "rephrased_query": "",
                "chitchat_response": "",
                "clarification_question": "",
                "operation_type": "",
                "anchor_id": None,
                "response_json": None,
                "diagram_type": "",
                "title": "",
                "graph": None,
                "action_summary": "",
            }
            
            # Convert doc_json to Synnoia format if present
            if doc_json:
                try:
                    print("Converting doc_json to Synnoia format...")
                    # Parse JSON string to dict first
                    doc_json_dict = json.loads(doc_json) if isinstance(doc_json, str) else doc_json
                    synnoia_doc = tiptap_to_synnoia(doc_json_dict)
                    # Convert back to JSON string for the state schema
                    initial_state["doc_json"] = json.dumps(synnoia_doc)
                    print("doc_json conversion complete")
                except Exception as e:
                    print(f"doc_json conversion error: {e}")
                    await websocket.send_json({"error": f"Failed to convert doc_json: {str(e)}"})
                    continue
            
            try:
                print("Invoking agent...")
                
                # Use invoke for now to ensure complete results
                final_result = synnoia_agent.invoke(initial_state)
                print("Agent invocation complete")
            except asyncio.TimeoutError:
                print("Agent execution timeout")
                await websocket.send_json({"error": "Agent execution timeout (5 minutes)"})
                continue
            except Exception as e:
                print(f"Agent execution error: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"error": f"Agent execution failed: {str(e)}"})
                continue
            
            # Prepare response with all new fields
            response_data = {
                "intent": final_result.get("intent", ""),
                "response": final_result.get("response", ""),
                "chitchat_response": final_result.get("chitchat_response", ""),
                "clarification_question": final_result.get("clarification_question", ""),
                "rephrased_query": final_result.get("rephrased_query", ""),
                "response_json": None,
                "operation_type": final_result.get("operation_type", ""),
                "anchor_id": final_result.get("anchor_id", None),
                "diagram_type": final_result.get("diagram_type", ""),
                "title": final_result.get("title", ""),
                "graph": final_result.get("graph", None),
                "action_summary": final_result.get("action_summary", ""),
            }
            
            # Convert response_json to TipTap if present
            if final_result.get("response_json"):
                try:
                    print("Converting response_json to TipTap...")
                    synnoia_data = final_result["response_json"].model_dump()
                    tiptap_data = synnoia_to_tiptap(synnoia_data)
                    response_data["response_json"] = tiptap_data
                    print("response_json conversion complete")
                except Exception as e:
                    print(f"response_json conversion error: {e}")
                    response_data["response_json"] = {"error": f"Failed to convert response_json: {str(e)}"}
            
            # Convert graph to XML if present
            if final_result.get("graph"):
                try:
                    print("Converting graph to XML...")
                    response_data["graph"] = final_result["graph"].to_xml()
                    print("graph conversion complete")
                except Exception as e:
                    print(f"graph conversion error: {e}")
                    response_data["graph"] = {"error": f"Failed to convert graph: {str(e)}"}
            
            # Send final response back to frontend
            print("Sending final response to client...")
            print(f"operation_type: {response_data['operation_type']}, anchor_id: {response_data['anchor_id']}, has_response_json: {bool(response_data['response_json'])}, has_graph: {bool(response_data['graph'])}")
            await websocket.send_json(response_data)
            print("Final response sent successfully")
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass



