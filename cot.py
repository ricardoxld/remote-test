from flask import jsonify, Flask
from flask_restful import Resource, Api, reqparse
import streamlit as st
from openai import OpenAI
import os
import json
import time
import requests



##https://zw.caseeder.cn/nlp/public/zkzw.do/downloadFile/%E5%B9%BF%E5%B7%9E%E5%B8%82%E4%BA%BA%E5%8A%9B%E8%B5%84%E6%BA%90%E5%92%8C%E7%A4%BE%E4%BC%9A%E4%BF%9D%E9%9A%9C%E5%B1%80%E5%85%B3%E4%BA%8E%E5%B9%BF%E5%B7%9E%E5%B8%822022%E5%B9%B4%E5%BA%A6%E7%A4%BE%E4%BC%9A%E4%BF%9D%E9%99%A9%E5%9F%BA%E9%87%91%E9%A2%84%E7%AE%97%E6%89%A7%E8%A1%8C%E6%83%85%E5%86%B5%E5%AE%A1%E8%AE%A1%E6%95%B4%E6%94%B9%E6%83%85%E5%86%B5%E7%9A%84%E5%85%AC%E5%91%8A.pdf?fileId=64df3da3-2da1-4635-8154-fb94647b4549&disposition=inline&
app = Flask(__name__)
api = Api(app)

def make_api_call(messages, is_final_answer=False):
    client = OpenAI(
        api_key = "sk-ftqwvyglkjlusikmrjylhyukwbmxjeoctskfriykqkttjbnf",
        base_url = "https://api.siliconflow.cn/v1",
    )

    for attempt in range(3): 
        try:
            response = client.chat.completions.create(
                model= "Qwen/Qwen2.5-7B-Instruct",  # Qwen/Qwen2.5-7B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct
                messages=messages,
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error",
                            "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """你是一位具有高级推理能力的人工智能专家助理。你的任务是对你的思维过程进行详细的、循序渐进的解释。对于每一步：
            1.提供一个清晰、简洁的标题，描述当前的推理阶段。
            2.对于每个步骤，提供一个描述你在该步骤中正在做什么的标题，以及详细的内容。
            3.决定是否需要另一个步骤，或者是否准备给出最终答案。
            4.以JSON格式响应，包含'title'（标题）、'content'（内容）和'next_action'（下一步动作，可以是'continue'或者'final_answer'）键。
            关键说明：
            - 尽可能使用多的推理步骤，至少3步。
            - 意识到你作为大型语言模型的局限性，清楚你能做什么和不能做什么。
            - 在推理过程中，探索替代答案。
            - 考虑到你可能会出错，并且如果你的推理是错误的，错误可能出现在什么地方。
            - 全面测试所有其他可能性。你可能是错的。
            - 当你说你正在重新检查时，真正地重新检查，并使用另一种方法进行检查。
            - 不要只是说你在重新检查。使用至少3种方法得出答案。采用最佳实践。
            """
        },
        {"role": "user", "content": prompt},
        {"role": "assistant","content": "谢谢！我现在会根据我的指示一步步进行思考，从分解问题开始，使用中文回答。"}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        time.sleep(3) # Simulate thinking time
        step_data = make_api_call(messages, 1000)
        print(step_data)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append((f"步骤 {step_count}: {step_data['title']}", step_data['content'], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data['next_action'] == 'final_answer':
            break

        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "请根据上述推理提供最终答案，使用中文回答。"})

    start_time = time.time()
    final_data = make_api_call(messages, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps_json = json.dumps(steps, ensure_ascii=False, indent=4)
    return steps_json

# 定义解析器
parser = reqparse.RequestParser()
parser.add_argument('url', type=str, required=True, location="json")

class CoT(Resource):
    def post(self):
        args = parser.parse_args()
        user_query = args['query']

        st.set_page_config(page_title="OpenAI Reasoning Chains", page_icon="🧠", layout="wide")

        st.title("通义千问推理链")

        st.markdown("""
        这是一个使用提示创建通义千问推理链的早期原型，旨在提高输出的准确性。
        """)

        try:
            st.write("生成响应中...")

            # Create empty elements to hold the generated text and total time
            response_container = st.empty()
            time_container = st.empty()

            result = generate_response(user_query)
            return result
        except Exception as e:
            # 其他异常处理
            error_message = str(e)
            return jsonify({"error": error_message})

api.add_resource(CoT, '/cot')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=15884)
