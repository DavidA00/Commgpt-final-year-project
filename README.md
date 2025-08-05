# CommGPT — A Teaching-Assistant LLM for Advanced Communication Systems

[**→ Model weights on Hugging Face**](https://huggingface.co/dabboud/Commgpt-3B)
---

## 1 · What is CommGPT?

*CommGPT* is a conversational language model built on **Qwen 2.5-3B-Instruct** and fine-tuned exclusively for the syllabus of **EECE 442 / Advanced Communication Systems** at the American University of Beirut.  Paired with a retrieval-augmented generation (RAG) backend during development, the model learned to answer domain-specific questions, walk through derivations, and offer study-friendly explanations in plain English.  While the full deployment stack involves indexing and GUI components, this repository contains only the **final weights** and the **comprehensive implementation report** that documents every design decision from data wrangling to fine-tuning.

---

## 2 · Why we built it: Motivation, Needs, Constraints

### Pedagogical motivation  
Communication-systems courses are concept-dense and mathematically demanding.  Students often need immediate feedback on derivations, parameter trade-offs or theoretical subtleties that rarely fit into standard office-hour windows.  CommGPT was conceived as a tireless assistant that can fill that gap, offering on-demand clarification without replacing the instructor’s deeper guidance.

### Practical needs  
To be useful in a rigorous engineering course, the assistant had to:

1. **Speak the language of the field**: notation for power spectral density, Nyquist criteria, matched filtering, QAM constellations.  
2. **Reference actual course material** so that answers align with lecture slides and textbook chapters.  
3. **Handle equation-heavy prompts** yet still output clear natural-language explanations suited to undergraduates.  

### Project constraints  
Several hard constraints shaped the implementation:

* **Data confidentiality** — Portions of the dataset were proprietary notes kindly provided by Prof. Ibrahim Abou Faycal; those pages could not be shared outside the research team.  
* **Compute budget** — The project had access to a single A100 GPU (80 GB) and limited cloud credits, ruling out gigantic models or multi-GPU sharding.  
* **Zero-code distribution** — Because the private notes cannot be redistributed and the RAG pipeline depends on them, the accompanying codebase remains internal.  Only the model weights and the report could be released publicly.  

---

## 3 · How it works (summary of the report)

Although the code itself is not part of this repository, the report (`Implementation.pdf`) walks through every stage:

* **Curated dataset**:  70 open-access PDFs plus select textbook excerpts were converted to Markdown via the Marker OCR/LLM pipeline, manually cleaned, chunked at ~750 words and enriched with keywords, difficulty scores and equation flags.  
* **Dual-index RAG**:  Each chunk received a compact summary; summaries formed one FAISS index while full text formed the second.  At inference time a query was embedded (Stella-400 M), retrieved with Maximum-Marginal-Relevance, reranked by a cross-encoder and finally passed to the generator.  
* **Curriculum fine-tuning**:  17 k QA pairs were generated in Easy → Medium → Hard stages.  Full fine-tuning of Qwen 2.5-3B via Unsloth on a single GPU brought the model from general instruction following to communication-engineering fluency without catastrophic forgetting.  
* **Evaluation & GUI**:  Informal MCQ benchmarks and a Gradio chat demo confirmed clear gains over the base model—especially on constellation-diagram reasoning and bandwidth/bit-rate trade-offs.

Every command, hyper-parameter choice, and failure mode (e.g. why Mathstral-7B was abandoned) is detailed in the PDF.

---

## 4 · What’s in this repository

| File | Description |
|------|-------------|
| `CommGPT-3B` folder (automatically pulled) | Int4 / bf16 weight shards + tokenizer configuration |
| `Implementation.pdf` | 70-page technical narrative covering data pipeline, RAG architecture, fine-tuning, evaluation and GUI snapshots |

No source code or datasets are included because they either reference proprietary material or require closed indices.

---

## 5 · Using the weights

Below is a minimal snippet that loads the model for pure text generation (without RAG context).  It still answers many EECE 442 questions from its internal knowledge learned during fine-tuning.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "dabboud/Commgpt-3B"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

chat = pipeline("text-generation", model=model, tokenizer=tok)
print(chat("Explain why the matched filter maximizes SNR in the presence of AWGN.",
           max_new_tokens=160)[0]["generated_text"])

```

If you wish to recreate the full RAG assistant, consult the pipeline diagrams and parameter tables in Chapters 3 and 4 of the report, then plug in your own domain documents.

## 6 · Acknowledgements
CommGPT exists thanks to:

Prof. Ibrahim Abou Faycal for granting access to proprietary textbook drafts and providing expert feedback.

Alibaba DAMO Academy for releasing Qwen 2.5 under an open license.

Marker developers for a remarkably accurate PDF→Markdown tool.

The authors of Stella-400 M and MS-MARCO Cross-Encoder for open-sourcing strong embedding models.

AUB’s HPC team and RunPod for the GPU hours that powered weeks of experimentation.

## 7 · License
The CommGPT weights and this README are released under the MIT License.
All third-party documents referenced during training retain their original licenses and are not redistributed here.

## 8 · Suggested citation

```latex
@misc{CommGPT2025,
  title   = {CommGPT: A Domain-Tuned Qwen 2.5-3B Model for Advanced Communication Systems},
  author  = {Dabboud, David and Abou Faycal, Ibrahim},
  note    = {Model and report available at \url{https://huggingface.co/dabboud/Commgpt-3B}},
  year    = {2025}
}
```

