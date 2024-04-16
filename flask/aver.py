import requests
from bs4 import BeautifulSoup

# URL de la página que quieres scrapear
url = "https://www.mindat.org/loc-2285.html"

# Realizar la solicitud GET a la página web
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Parsear el contenido HTML usando BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Encontrar todos los elementos <a> con la clase 'mr10'
    mineral_links = soup.find_all("a", class_="mr10")
    
    # Extraer los nombres de los minerales
    mineral_names = [link.get_text() for link in mineral_links]
    
    # Imprimir los nombres de los minerales
    for name in mineral_names:
        print(name)
else:
    print("Error al acceder a la página:", response.status_code)