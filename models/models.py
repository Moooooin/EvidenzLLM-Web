"""
Query Classifier Model
Copied exactly from notebook section "2. Modeldefinitionen"
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from safetensors.torch import load_file as load_safetensors


# Label mapping from notebook
label_map = {"factual_lookup": 0, "explanation": 1, "reasoning": 2, "calculation": 3}


class QueryClassifier(nn.Module):

  """
  Query-Klassifikator: Transformer-Encoder + linearer Klassifikationskopf.
  Extrahiert CLS-Token, wendet Dropout an und gibt Logits zurück.
  Berechnet optional Kreuzentropie-Loss, wenn Labels übergeben werden.
  """

  def __init__(self, num_labels=4, model_name="microsoft/deberta-base"):
    super().__init__()

    # Transformer-Encoder laden (z.B. DeBERTa)
    self.encoder = AutoModel.from_pretrained(model_name)
    hidden = self.encoder.config.hidden_size

    # Dropout-Layer zur Regularisierung
    self.dropout = nn.Dropout(0.2)

    # Linearer Klassifikationskopf für Label Klassen
    self.classifier = nn.Linear(hidden, num_labels)

  def forward(self, input_ids, attention_mask, labels=None):

    # Encoder-Ausgabe
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    # CLS-Token extrahieren (repräsentiert ganze Sequenz)
    cls_emb = outputs.last_hidden_state[:, 0, :]

    # Dropout anwenden
    x = self.dropout(cls_emb)

    # Logits durch Klassifikationskopf erzeugen
    logits = self.classifier(x)

    loss = None
    # Kreuzentropie-Loss berechnen, falls Labels vorhanden
    if labels is not None:
      loss = F.cross_entropy(logits, labels)

    return {"loss": loss, "logits": logits}



def load_query_classifier(model_load_path, device):
  """
  Load QueryClassifier model from safetensors or pytorch_model.bin
  Copied exactly from notebook section "### Modell laden"
  
  Args:
    model_load_path: Path to the model directory
    device: torch device (cuda or cpu)
    
  Returns:
    Loaded QueryClassifier model in eval mode
  """
  print(f"Lade Modell von: {model_load_path}")
  try:
    # Neue Modellinstanz erstellen
    loaded_model = QueryClassifier(num_labels=4)
    loaded_model.to(device)

    # Pfade zu den State Dictionary Dateien
    safetensors_path = os.path.join(model_load_path, 'model.safetensors')
    pytorch_bin_path = os.path.join(model_load_path, 'pytorch_model.bin')

    state_dict = None

    # Von safetensors laden
    if os.path.exists(safetensors_path):
      print(f"Lade State Dictionary von: {safetensors_path}")
      state_dict = load_safetensors(safetensors_path)
    # Alternativ über pytorch_model.bin laden
    elif os.path.exists(pytorch_bin_path):
      print(f"Lade State Dictionary von: {pytorch_bin_path}")
      state_dict = torch.load(pytorch_bin_path, map_location=device)
    else:
      raise FileNotFoundError(f"Keine Modelldatei (model.safetensors oder pytorch_model.bin) unter {model_load_path} gefunden.")

    # Lade den State Dictionary in das Modell
    if state_dict is not None:
      loaded_model.load_state_dict(state_dict)
      loaded_model.eval() # Evaluationsmodus
      print("Modell erfolgreich geladen.")
    else:
      print("Konnte State Dictionary nicht laden.")

  except FileNotFoundError as e:
    print(f"Fehler beim Laden des Modells: {e}")
    raise
  except NameError:
    print("QueryClassifier Klasse nicht gefunden. Stelle sicher, dass die Modelldefinition geladen wurde.")
    raise
  except Exception as e:
    print(f"Ein unerwarteter Fehler beim Laden des Modells ist aufgetreten: {e}")
    raise
  
  return loaded_model
