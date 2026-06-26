"""
Generates Python reference embeddings for the Rhobots validation section.

Run once from the repo root:
    python3 paper/validate_embeddings.py

Requires:
    pip install sentence-transformers pyreadr numpy

Writes:
    paper/py_embeddings.rds   (float32 matrix, N x 384)
    paper/val_sentences.rds   (character vector of N sentences)
"""

import numpy as np
import pyreadr
import pandas as pd
from sentence_transformers import SentenceTransformer

VAL_SENTENCES = [
    # Energy
    "Solar photovoltaic panels have become the cheapest electricity source.",
    "Wind turbines generate electricity by converting kinetic energy.",
    "Battery storage is essential for integrating renewable energy into grids.",
    "Offshore wind farms can produce more power than onshore installations.",
    "Hydrogen produced from renewable electricity can decarbonize industry.",
    "Smart grids balance supply and demand across distributed energy systems.",
    "Nuclear power produces low-carbon baseload electricity continuously.",
    "Geothermal energy harnesses heat from the Earth's interior.",
    "Energy efficiency measures reduce demand without sacrificing services.",
    "Bioenergy with carbon capture can achieve negative emissions.",
    # Water
    "Freshwater availability is threatened by climate change and over-extraction.",
    "Groundwater depletion in agricultural regions poses long-term food risks.",
    "Wastewater treatment and reuse are central to circular water management.",
    "River basin governance must balance upstream and downstream water users.",
    "Desalination provides drinking water in arid coastal regions.",
    "Wetlands filter pollutants and buffer floods in river catchments.",
    "Irrigation efficiency improvements can significantly reduce agricultural water use.",
    "Virtual water trade redistributes water-intensive production globally.",
    "Transboundary aquifer agreements are necessary for shared groundwater resources.",
    "Urban water infrastructure is aging and requires substantial reinvestment.",
    # Biodiversity
    "Habitat fragmentation is the leading driver of terrestrial species loss.",
    "Marine protected areas help rebuild fish stocks and restore reef systems.",
    "Pollinator decline threatens crop yields and natural ecosystem functioning.",
    "Rewilding programs reintroduce apex predators to restore trophic cascades.",
    "Invasive species alter community composition and outcompete native organisms.",
    "Seed banks preserve genetic diversity of crop wild relatives.",
    "Corridor networks connect fragmented habitats to support wildlife movement.",
    "Ocean acidification threatens calcifying organisms such as corals and molluscs.",
    "Fire regimes shape savanna and chaparral biodiversity dynamics.",
    "Phenological mismatches reduce reproductive success of migratory species.",
    # Climate
    "Global mean temperature has risen above pre-industrial levels.",
    "Carbon capture and storage technologies remain expensive and unproven at scale.",
    "Permafrost thaw releases methane, accelerating Arctic warming feedbacks.",
    "Urban heat islands amplify extreme heat events in densely populated cities.",
    "Sea-level rise threatens coastal infrastructure and displaces communities.",
    "Aerosol emissions mask a fraction of greenhouse gas warming.",
    "Clouds represent the largest source of uncertainty in climate projections.",
    "El Niño–Southern Oscillation modulates global precipitation patterns.",
    "Stratospheric ozone depletion and climate change interact in complex ways.",
    "Attribution science links individual weather events to climate change.",
    # Governance
    "Multilateral environmental agreements require domestic enforcement mechanisms.",
    "Carbon pricing instruments differ in their distributional consequences.",
    "Civil society organizations play a key role in environmental accountability.",
    "Science-policy interfaces are critical for evidence-based climate governance.",
    "Nationally Determined Contributions set emissions targets under the Paris Agreement.",
    "Environmental justice links pollution exposure to socioeconomic inequality.",
    "Corporate sustainability reporting frameworks are converging internationally.",
    "Indigenous land rights are central to effective forest conservation.",
    "Regulatory capture undermines the independence of environmental agencies.",
    "Loss and damage mechanisms address climate harms in vulnerable nations.",
]

print(f"Embedding {len(VAL_SENTENCES)} sentences...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    VAL_SENTENCES,
    normalize_embeddings=True,
    convert_to_numpy=True,
    precision="float32",
)

print(f"Shape: {embeddings.shape}")
print(f"L2 norms (all should be 1.0): {np.round(np.linalg.norm(embeddings, axis=1), 6)}")

pyreadr.write_rds("paper/py_embeddings.rds", pd.DataFrame(embeddings))
pyreadr.write_rds("paper/val_sentences.rds",
                  pd.DataFrame({"sentence": VAL_SENTENCES}))

print("Saved paper/py_embeddings.rds and paper/val_sentences.rds")
