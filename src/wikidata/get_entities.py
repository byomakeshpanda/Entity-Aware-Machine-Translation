import requests

def get_wikidata_label(entity_id, preferred_langs=['en']):
    """
    Fetches labels for an entity from Wikidata in preferred languages.
    
    Args:
        entity_id (str): The Wikidata entity ID (e.g., "Q42").
        preferred_langs (list): List of language codes (e.g., ["fr", "en"]).
    
    Returns:
        dict: A dictionary with language codes as keys and entity labels as values.
    """
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&format=json&props=labels"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        entity_data = data.get("entities", {}).get(entity_id, {})

        labels = entity_data.get("labels", {})
        result = {lang: labels[lang]["value"] for lang in preferred_langs if lang in labels}

        return result if result else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Wikidata data for {entity_id}: {e}")
        return None
