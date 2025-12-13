import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import requests
import json
import re
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Set
import time


class SketchfabGLTFIntegrator:
    """
    Integrates Sketchfab 3D models into A-Frame scenes when the model
    doesn't know how to handle certain objects.
    """
    
    # Common A-Frame environment preset objects that the model already handles
    KNOWN_OBJECTS = {
        'sky', 'ground', 'trees', 'mountain', 'mountains', 'forest', 'desert', 
        'ocean', 'pyramids', 'egypt', 'snow', 'dust', 'rain', 'particle', 'light',
        'sun', 'moon', 'stars', 'clouds', 'fog', 'box', 'sphere', 'cylinder',
        'cone', 'plane', 'torus', 'ring'
    }
    
    def __init__(self, api_token: str, download_dir: str = "./downloaded_models"):
        self.api_token = api_token
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.base_url = "https://api.sketchfab.com/v3"
        
    def extract_objects_from_prompt(self, prompt: str) -> Set[str]:
        # Common nouns that might be 3D objects
        words = re.findall(r'\b[a-z]+\b', prompt.lower())
        
        # Filter out common stop words and keep potential objects
        stop_words = {'a', 'an', 'the', 'with', 'at', 'and', 'or', 'in', 'on', 
                     'scene', 'light', 'lighting', 'heavy', 'some', 'many'}
        
        potential_objects = {word for word in words if word not in stop_words}
        
        return potential_objects
    
    def find_unknown_objects(self, prompt: str) -> Set[str]:
        all_objects = self.extract_objects_from_prompt(prompt)
        unknown = all_objects - self.KNOWN_OBJECTS
        
        # Filter out very common words that are likely not 3D objects
        filtered_unknown = {obj for obj in unknown if len(obj) > 3}
        
        return filtered_unknown
    
    def search_model(self, query: str, max_results: int = 5) -> List[Dict]:
        search_url = f"{self.base_url}/search"
        params = {
            "type": "models",
            "q": query,
            "downloadable": True,  # Only get downloadable models
            "count": max_results,
            "sort_by": "-likeCount"  # Sort by popularity
        }
        
        headers = {
            "Authorization": f"Token {self.api_token}"
        }
        
        try:
            response = requests.get(search_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for result in data.get("results", []):
                if result.get("isDownloadable", False):
                    models.append({
                        "uid": result["uid"],
                        "name": result["name"],
                        "thumbnail": result.get("thumbnails", {}).get("images", [{}])[0].get("url"),
                        "viewerUrl": result.get("viewerUrl", "")
                    })
            
            return models
        except requests.exceptions.RequestException as e:
            print(f"Error searching for '{query}': {e}")
            return []
    
    def get_download_url(self, model_uid: str) -> Optional[str]:
        download_url = f"{self.base_url}/models/{model_uid}/download"
        headers = {
            "Authorization": f"Token {self.api_token}"
        }
        
        try:
            response = requests.get(download_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            gltf_data = data.get("gltf", {})
            if gltf_data and "url" in gltf_data:
                return gltf_data["url"]
            
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting download URL for model {model_uid}: {e}")
            return None
    
    def download_model(self, model_uid: str, object_name: str) -> Optional[str]:
        download_url = self.get_download_url(model_uid)
        if not download_url:
            print(f"Could not get download URL for {object_name}")
            return None
        
        safe_name = re.sub(r'[^\w\-]', '_', object_name)
        zip_path = self.download_dir / f"{safe_name}.zip"
        extract_dir = self.download_dir / safe_name
        
        try:
            print(f"Downloading {object_name} from {download_url}...")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Save the zip file
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded to {zip_path}, extracting...")
            
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the .gltf or .glb file
            gltf_files = list(extract_dir.glob("**/*.gltf")) + list(extract_dir.glob("**/*.glb"))
            
            if gltf_files:
                gltf_path = str(gltf_files[0])
                print(f"Successfully extracted glTF to {gltf_path}")
                
                zip_path.unlink()
                
                return gltf_path
            else:
                print(f"No glTF file found in downloaded archive for {object_name}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            return None
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip file: {e}")
            return None
    
    def inject_gltf_into_html(self, html_content: str, object_assets: Dict[str, str]) -> str:
        """
        Inject glTF models into A-Frame HTML.
        
        Args:
            html_content: Original A-Frame HTML
            object_assets: Dict mapping object names to local glTF file paths
            
        Returns:
            Modified HTML with glTF models included
        """
        if not object_assets:
            return html_content
        
        # Build <a-assets> section
        assets_html = "    <a-assets>\n"
        entity_html = ""
        
        for i, (obj_name, file_path) in enumerate(object_assets.items()):
            asset_id = f"{obj_name}_{i}"
            assets_html += f'      <a-asset-item id="{asset_id}" src="{file_path}"></a-asset-item>\n'
            
            x_pos = (i % 3 - 1) * 5  # -5, 0, 5
            z_pos = -10 - (i // 3) * 5
            entity_html += f'      <a-entity gltf-model="#{asset_id}" position="{x_pos} 0 {z_pos}" scale="1 1 1"></a-entity>\n'
        
        assets_html += "    </a-assets>\n"
        
        scene_pattern = r'(<a-scene[^>]*>)'
        
        if re.search(scene_pattern, html_content):
            # Inject assets right after <a-scene>
            html_content = re.sub(
                scene_pattern,
                r'\1\n' + assets_html + entity_html,
                html_content,
                count=1
            )
        else:
            print("Warning: Could not find <a-scene> tag in HTML")
        
        return html_content
    
    def process_prompt_and_enhance_html(
        self, 
        prompt: str, 
        generated_html: str,
        auto_download: bool = True
    ) -> tuple[str, List[Dict]]:
        """
        Main method: Find unknown objects, search for models, and enhance HTML.
        
        Args:
            prompt: User's scene description
            generated_html: HTML generated by the fine-tuned model
            auto_download: If True, automatically download the first match for each object
            
        Returns:
            Tuple of (enhanced_html, list of model info dicts)
        """
        unknown_objects = self.find_unknown_objects(prompt)
        
        if not unknown_objects:
            print("No unknown objects found in prompt")
            return generated_html, []
        
        print(f"Found unknown objects: {unknown_objects}")
        
        downloaded_models = {}
        all_model_info = []
        
        for obj in unknown_objects:
            print(f"\nSearching for '{obj}'...")
            models = self.search_model(obj, max_results=3)
            
            if not models:
                print(f"No models found for '{obj}'")
                continue
            
            print(f"Found {len(models)} models for '{obj}':")
            for i, model in enumerate(models):
                print(f"  {i+1}. {model['name']} (uid: {model['uid']})")
            
            all_model_info.extend(models)
            
            if auto_download and models:
                
                best_model = models[0]
                print(f"Auto-downloading: {best_model['name']}")
                
                file_path = self.download_model(best_model['uid'], obj)
                if file_path:
                    downloaded_models[obj] = file_path
                
                time.sleep(1)
        
        enhanced_html = self.inject_gltf_into_html(generated_html, downloaded_models)
        
        return enhanced_html, all_model_info




if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    API_TOKEN = "b104e9c4ece54e9dbc2e418e27f2eb3b"
    
    base_model = "Qwen/Qwen3-1.7B-Base"
    adapter_path = "./qwen_sft_vr_run2/final"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"  
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    
    integrator = SketchfabGLTFIntegrator(api_token=API_TOKEN)
    
    prompt = "A scene with a sports car and a skyscraper"
    inputs = tokenizer(prompt + "\nHTML_START\n", return_tensors="pt").to("cuda")
    
    print("Generating HTML...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    generated_html = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    html_match = re.search(r'HTML_START\n(.*?)HTML_END', generated_html, re.DOTALL)
    if html_match:
        html_content = html_match.group(1)
        
        enhanced_html, models = integrator.process_prompt_and_enhance_html(
            prompt, 
            html_content,
            auto_download=True
        )
        
        with open("final_scene.html", "w") as f:
            f.write(enhanced_html)
        
        print(f"\nEnhanced HTML saved to final_scene.html")
        print(f"Total models found: {len(models)}")
    else:
        print("Could not extract HTML from generated output") 