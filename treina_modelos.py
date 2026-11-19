from ultralytics import YOLO
import os
import shutil
import gc
import torch

def treina_modelos(path, yolo_model, name):
    lista = os.listdir(path)
    
    for i, ordem in enumerate(lista):
        print(f"[ {i} | {len(lista)} ] - Iniciando treino para: {ordem}")
        
        model = YOLO(yolo_model)

        model.train(
            data=os.path.join(path, ordem),
            epochs=100,
            imgsz=224,
            batch=64,
            workers=10,
            device=0,
            amp=True,
            cache=False,
        )
        # ATENÇÃO: Ao executar essa função, garanta que a pasta runs não exista para mover corretamente
        
        print("Movendo resultados")
        destino = f"/home/bradas/Documentos/github/ml-engineering/modelos/modelos-hierarquico-{name}/yolo-{name}-{ordem}"
        os.makedirs(destino, exist_ok=True)
        
        origem_runs = "/home/bradas/Documentos/github/ml-engineering/runs"
        origem_weights = "/home/bradas/Documentos/github/ml-engineering/weights"
        
        if os.path.exists(origem_runs):
            shutil.move(origem_runs, destino + "/runs")
        if os.path.exists(origem_weights):
            shutil.move(origem_weights, destino + "/weights")
            
        del model
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"Memória liberada após o treino de {ordem}\n")

treina_modelos("/home/bradas/Downloads/insetos/familia/familia_yolo_hierarquico", 'yolo26n-cls.pt', "nano")