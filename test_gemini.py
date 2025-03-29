import google.generativeai as genai

gemini_api_key = ""
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)