
import openai
import os
import json

class gpt_api:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_vlm_output(self, rgb_image, user_command):
        prompt = f"""
You are controlling a Sawyer robot arm.

Given a user command, output a JSON object with **TWO fields**:

- "actions": a list of one or more actions, in order, from: [move_to, open_gripper, close_gripper, lift(delta_z)]
  When the user says multiple things (e.g. "move forward and open gripper"), put multiple actions in the list.
- "position": (x, y) in meters for move_to. Use this for the first move_to if there is one. Assume z=0.6. Forward = +x from center (0.6, 0), so e.g. 5 cm forward = [0.65, 0.0], 10 cm = [0.7, 0.0]. Right = -y, left = +y.

---
**Examples:**

User command: "move forward by 10 centimeters"
Output: {{ "actions": ["move_to"], "position": [0.7, 0.0] }}

User command: "move right by 20 centimeters"
Output: {{ "actions": ["move_to"], "position": [0.6, -0.2] }}

User command: "move to position (0.5, 0.1)"
Output: {{ "actions": ["move_to"], "position": [0.5, 0.1] }}

User command: "move forward by 5 cm and open gripper"
Output: {{ "actions": ["move_to", "open_gripper"], "position": [0.65, 0.0] }}

User command: "close gripper then lift by 0.1"
Output: {{ "actions": ["close_gripper", "lift(0.1)"], "position": [0.6, 0.0] }}

---
Only output valid JSON — no extra words.

User command: "{user_command}"
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            message_content = response.choices[0].message.content
            
            # Check if response is empty
            if not message_content or message_content.strip() == "":
                print(f"GPT API error: Empty response from API")
                return {
                    "actions": [],
                    "position": [0.6, 0.0]
                }
            
            # Try to extract JSON if there's extra text
            message_content = message_content.strip()
            
            # Find JSON object in response (in case there's extra text)
            start_idx = message_content.find('{')
            end_idx = message_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = message_content[start_idx:end_idx]
            else:
                json_str = message_content
            
            output = json.loads(json_str)
            return output

        except json.JSONDecodeError as e:
            print(f"GPT API JSON parsing error: {e}")
            print(f"Response was: {message_content if 'message_content' in locals() else 'N/A'}")
            return {
                "actions": [],
                "position": [0.6, 0.0]
            }
        except Exception as e:
            print(f"GPT API error: {e}")
            return {
                "actions": [],
                "position": [0.6, 0.0]
            }

