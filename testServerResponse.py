import requests

url = "http://localhost:8000/predict/"
file_path = r"C:\Users\sadfi\OneDrive\Bureau\AI courses & Books\coursera IBM AI\Ai capstone Test\concrete_data_week4\valid\positive\15003_1.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())