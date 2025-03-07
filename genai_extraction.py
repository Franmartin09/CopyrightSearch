import os
from datetime import datetime
import json
from google import genai
from dotenv import load_dotenv
import tiktoken
import time

# Cargar el archivo .env y forzar la sobreescritura de las variables de entorno
load_dotenv(override=True)

def model_output(text, client):
    prompt = f"""
    Analyze the following text and extract all lines that seems a copyright or contribution statements. I need to declarate for each statement, the years, authors, and any associated emails (if available).

    ### Input Text:
    '''
    {text}
    '''

    ### Example of Output Format:
    Copyright (c) [Years] [Author1] [Email1], [Author2] [Email2], ...,
    ...
    

    ### Guidelines:
    - Extract all copyright or contribution statements present in the text.
    - For each copyright or contribution statement, extract the years, authors, and contributors.
    - If an email is available, include it; otherwise, omit it (do not insert placeholders).
    - Each copyright or contribution statement should be listed on a separate line within the output list.
    - Ensure the output follows the exact format than the input text without any extra text or explanations.
    """
   
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    return response.text


def predict_results(json_file, client, output_json):
    # Cargar datos del JSON de entrada
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    count_processed = 0
    if os.path.exists('checkpoint.json'):
        with open('checkpoint.json', "r", encoding="utf-8") as file:
            checkpoint_data = json.load(file)
            count_processed = checkpoint_data.get("count", 0)

    existing_results = []
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as file:
            try:
                existing_results = json.load(file)
            except json.JSONDecodeError:
                existing_results = []

    results = existing_results
    
    total = len(data)
    try:
        for i, comment in enumerate(data):
            if i < count_processed:
                    continue  # Saltar elementos ya procesados
        
            result = model_output(comment['comments'], client)

            # Agregar resultado junto con el archivo
            results.append({
                "file": comment['file'],
                "comment": comment['comments'],
                "result": result
            })

            time.sleep(5)
            # Guardar checkpoint periódicamente
            with open('checkpoint.json', "w", encoding="utf-8") as file:
                json.dump({"count": i}, file, ensure_ascii=False, indent=4)

            progress = (i + 1) / total * 100
            print(f"\rProgreso: {i + 1}/{total} comentarios procesados ({progress:.2f}% completado)", end='\r')
            
        # Guardar todos los resultados en el archivo JSON
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

        if os.path.exists('checkpoint.json'):
            os.remove('checkpoint.json')

    except Exception as e:
        print("\nInterruption detected. Saving progress...")
        print(f"\nException: {e}")

        # Guardar checkpoint antes de salir
        with open('checkpoint.json', "w", encoding="utf-8") as file:
            json.dump({"count": i}, file, ensure_ascii=False, indent=4)

        # Guardar resultados parciales
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
    
    except KeyboardInterrupt:
        print("\nInterruption detected. Saving progress...")

        # Guardar checkpoint antes de salir
        with open('checkpoint.json', "w", encoding="utf-8") as file:
            json.dump({"count": i}, file, ensure_ascii=False, indent=4)

        # Guardar resultados parciales
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

key=os.getenv("GEMINI_API_KEY")
print(key)
client = genai.Client(api_key=key)
predict_results(os.getenv('DATA_PATH'), client, os.getenv('OUTPUT_JSON'))
   