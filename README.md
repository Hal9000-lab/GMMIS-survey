# Generalist Models in Medical Image Segmentation: A Survey and Performance Comparison with Task-Specific Approaches

![Taxonomy on generalist models for medical image segmentation](images/taxonomy.png)

## SAM-based models

| Model | Publication | Classification | Code |
|-------|-------------|----------------|------|
|---| [Segment anything model for medical image analysis: An experimental study](https://www.sciencedirect.com/science/article/pii/S1361841523001780?via%3Dihub)| Zero-shot of SAM | [Code](https://github.com/mazurowski-lab/segment-anything-medical-evaluation)|
|---| [Segment anything model for medical images?](https://www.sciencedirect.com/science/article/pii/S1361841523003213) | Zero-shot of SAM | [Code](https://github.com/yuhoo0302/Segment-Anything-Model-for-Medical-Images)|
| SAM-MPA | [SAM-MPA: Applying SAM to Few-shot Medical Image Segmentation using Mask Propagation and Auto-prompting](https://arxiv.org/abs/2411.17363)| Few-shot of SAM | --- |
| SAM-Med2D | [SAM-Med2D](https://arxiv.org/abs/2308.16184)| Full fine-tuning of SAM | [Code](https://github.com/OpenGVLab/SAM-Med2D)|
| MedSAM | [Segment Anything in Medical Images](https://arxiv.org/abs/2304.12306)| Full fine-tuning of SAM | [Code](https://github.com/bowang-lab/MedSAM)|
| SAMed | [Customized Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.13785)| FEFT of SAM | [Code](https://github.com/hitachinsk/SAMed)|
| FLAP-SAM | [A Federated Learning-Friendly Approach for Parameter-Efficient Fine-Tuning of SAM in 3D Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-77610-6_21)| FEFT of SAM | [Code](https://github.com/BioMedIA-MBZUAI/FLAP-SAM)|
| Cheap Lunch SAM | [Cheap Lunch for Medical Image Segmentation by Fine-tuning SAM on Few Exemplars](https://arxiv.org/abs/2308.14133)| FEFT of SAM | --- |
| DeSAM | [DeSAM: Decoupled Segment Anything Model for Generalizable Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_48)| Modifications to SAM architecture | [Code](https://github.com/yifangao112/DeSAM)|
| SAM-Med3D | [SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images](https://arxiv.org/abs/2310.15161)| Modifications to SAM architecture | [Code](https://github.com/uni-medical/SAM-Med3D)|
| SAM3D | [SAM3D: Segment Anything Model in Volumetric Medical Images](https://arxiv.org/abs/2309.03493)| Modifications to SAM architecture | [Code](https://github.com/UARK-AICV/SAM3D)|
| 3DMedSAM | [Volumetric medical image segmentation via fully 3D adaptation of Segment Anything Model](https://www.sciencedirect.com/science/article/pii/S0208521624000846?via%3Dihub)| Adapters for SAM | --- |
| MedSA | [Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.12620)| Adapters for SAM | [Code](https://github.com/SuperMedIntel/Medical-SAM-Adapter)|
| MA-SAM | [MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation](https://arxiv.org/abs/2309.08842)| Adapters for SAM | [Code](https://github.com/cchen-cc/MA-SAM)|
| LeSAM | [LeSAM: Adapt Segment Anything Model for Medical Lesion Segmentation](https://ieeexplore.ieee.org/document/10540651)| Adapters for SAM | --- |
| 3DSAM-adapter | [3DSAM-adapter: Holistic adaptation of SAM from 2D to 3D for promptable tumor segmentation](https://arxiv.org/abs/2306.13465)| Adapters for SAM | [Code](https://github.com/med-air/3DSAM-adapter)|
| TP Mamba | [Tri-Plane Mamba: Efficiently Adapting Segment Anything Model for 3D Medical Images](https://link.springer.com/chapter/10.1007/978-3-031-72114-4_61)| Adapters for SAM | [Code](https://github.com/xmed-lab/TP-Mamba)|
| EMedSAM | [An efficient segment anything model for the segmentation of medical images](https://www.nature.com/articles/s41598-024-70288-8)| Adapters for SAM | --- |
| SPA | [SPA: Leveraging the SAM with Spatial Priors Adapter for Enhanced Medical Image Segmentation](https://ieeexplore.ieee.org/document/10829779)| Adapters for SAM | --- |
| M-SAM | [Mask-Enhanced Segment Anything Model for Tumor Lesion Semantic Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-72111-3_38)| Adapters for SAM | [Code](https://github.com/nanase1025/M-SAM)|
| SAMM | [SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM](https://arxiv.org/abs/2304.05622)| SAM for medical annotation | [Code](https://github.com/bingogome/samm)|
| SAMMed | [SAMMed: A medical image annotation framework based on large vision model](https://arxiv.org/abs/2307.05617)| SAM for medical annotation | --- |
| KnowSAM | [Learnable Prompting SAM-induced Knowledge Distillation for Semi-supervised Medical Image Segmentation](https://arxiv.org/abs/2412.13742)| Other SAM implementations| [Code](https://github.com/taozh2017/KnowSAM)|
| SFR SAM | [Stitching, Fine-tuning, Re-training: A SAM-enabled Framework for Semi-supervised 3D Medical Image Segmentation](https://arxiv.org/abs/2403.11229)| Other SAM implementations| [Code](https://github.com/ShumengLI/SFR)|
| MedLSAM | [MedLSAM: Localize and Segment Anything Model for 3D CT Images](https://arxiv.org/abs/2306.14752)| Other SAM implementations| [Code](https://github.com/openmedlab/MedLSAM)|

## SAM 2-based models
| Model | Publication | Classification | Code |
|-------|-------------|----------------|------|
| --- | [Interactive 3D Medical Image Segmentation with SAM 2](https://arxiv.org/abs/2408.02635)| Zero-shot of SAM 2 | [Code](https://github.com/Chuyun-Shen/SAM_2_Medical_3D)|
| ---  | [Zero-shot 3D Segmentation of Abdominal Organs in CT Scans Using Segment Anything Model 2: Adapting Video Tracking Capabilities for 3D Medical Imaging](https://arxiv.org/abs/2408.06170)| Zero-shot of SAM 2 | --- |
| ---  | [Segment anything model 2: an application to 2D and 3D medical images](https://arxiv.org/abs/2408.00756)| Zero-shot of SAM 2 | [Code](https://github.com/mazurowski-lab/segment-anything2-medical-evaluation)|
| ---  | [Is SAM 2 Better than SAM in Medical Image Segmentation?](https://arxiv.org/abs/2408.04212)| Zero-shot of SAM 2 | --- |
| MedSAM2 | [MedSAM2: Segment Anything in 3D Medical Images and Videos](https://arxiv.org/abs/2504.03600)| Fine-tuning of SAM 2 | [Code](https://github.com/bowang-lab/MedSAM2)|
| Biomedical SAM 2 | [Biomedical SAM 2: Segment Anything in Biomedical Images and Videos](https://arxiv.org/abs/2408.03286)| Fine-tuning of SAM 2 | [Code](https://github.com/ZhilingYan/Biomedical-SAM-2)|
| --- | [Segment Anything in Medical Images and Videos: Benchmark and Deployment](https://arxiv.org/abs/2408.03322)| Other applications of SAM 2 | [Code](https://github.com/bowang-lab/MedSAM)|
| Medical SAM 2| [Medical SAM 2: Segment medical images as video via Segment Anything Model 2](https://arxiv.org/abs/2408.00874)| Other applications of SAM 2 | [Code](https://supermedintel.github.io/Medical-SAM2/)|

## Other models trained only on images
| Model | Publication | Code |
|-------|-------------|------|
| UniMiSS| [UniMiSS: Universal Medical Self-supervised Learning via Breaking Dimensionality Barrier](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_33)| [Code](https://github.com/YtongXie/UniMiSS-code)|
| Med3D| [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625)| [Code](https://github.com/Tencent/MedicalNet)|
| SMIT | [Self-supervised 3D anatomy segmentation using self-distilled masked image transformer (SMIT)](https://arxiv.org/abs/2205.10342)| [Code]( https://github.com/TheVeeraraghavan-Lab/SMIT)|
| Hermes | [Training Like a Medical Resident: Context-Prior Learning Toward Universal Medical Image Segmentation](https://arxiv.org/abs/2306.02416)| [Code](https://github.com/yhygao/universal-medical-image-segmentation)|
| MIS-FM | [MIS-FM: 3D Medical Image Segmentation using Foundation Models Pretrained on a Large-Scale Unannotated Dataset](https://arxiv.org/abs/2306.16925)| [Code](https://github.com/openmedlab/MIS-FM)|
| MoME | [A Foundation Model for Brain Lesion Segmentation with Mixture of Modality Experts](https://arxiv.org/abs/2405.10246)| [Code](https://github.com/ZhangxinruBIT/MoME)|
| IMIS-Net | [Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](https://arxiv.org/abs/2411.12814)| [Code](https://github.com/uni-medical/IMIS-Bench)|
| UniverSeg | [UniverSeg: Universal Medical Image Segmentation](https://arxiv.org/abs/2304.06131)| | [Code](https://universeg.csail.mit.edu/)|
| STU-Net | [STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training](https://arxiv.org/abs/2304.06716)| [Code](https://github.com/Ziyan-Huang/STU-Net)|
| MultiTalent | [MultiTalent: A Multi-dataset Approach to Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-43898-1_62)| [Code](https://github.com/MIC-DKFZ/MultiTalent)|
| UniSeg| [UniSeg: A Prompt-Driven Universal Segmentation Model as Well as A Strong Representation Learner](https://link.springer.com/chapter/10.1007/978-3-031-43898-1_49)| [Code](https://github.com/yeerwen/UniSeg)|
|UniSeg33A | [Universal Segmentation of 33 Anatomies](https://arxiv.org/abs/2203.02098) | --- |
| BrainSegFounder | [BrainSegFounder: Towards 3D Foundation Models for Neuroimage Segmentation](https://arxiv.org/abs/2406.10395)| [Code](https://github.com/lab-smile/BrainSegFounder)|
| One-Prompt | [One-Prompt to Segment All Medical Images](https://arxiv.org/abs/2305.10300)|  [Code](https://github.com/SuperMedIntel/one-prompt)|
| DeSD | [DeSD: Self-Supervised Learning with Deep Self-Distillation for 3D Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_52)| [Code](https://github.com/yeerwen/DeSD)|
| DoDNet | [DoDNet: Learning to segment multi-organ and tumors from multiple partially labeled datasets](https://arxiv.org/abs/2011.10217)| [Code](https://github.com/aim-uofa/partially-labelled)|
| Disruptive Autoencoders  | [Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training](https://arxiv.org/abs/2307.16896)| [Code](https://github.com/Project-MONAI/research-contributions/tree/main/DAE)|

## Other models trained only on both images and text
| Model | Publication | Code |
|-------|-------------|------|
| BiomedParse| [BiomedParse: a biomedical foundation model for image parsing of everything everywhere all at once](https://arxiv.org/abs/2405.12971)| [Code](https://microsoft.github.io/BiomedParse/)|
| CLIP-Driven Universal Model | [CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection](https://arxiv.org/abs/2301.00785)| [Code](https://github.com/ljwztc/CLIP-Driven-Universal-Model)|
| Merlin | [Merlin: A Vision Language Foundation Model for 3D Computed Tomography](https://arxiv.org/abs/2406.06512)| [Code](https://github.com/StanfordMIMI/Merlin)|
| SAT | [One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompts](https://arxiv.org/abs/2312.17183)| [Code](https://github.com/zhaoziheng/SAT)|
| SegVol | [SegVol: Universal and Interactive Volumetric Medical Image Segmentation](https://arxiv.org/abs/2311.13385)| [Code]( https://github.com/BAAI-DCAI/SegVol)|
| PCNet | [PCNet: Prior Category Network for CT Universal Segmentation Model](https://ieeexplore.ieee.org/document/10510478 )| [Code](https://github.com/PKU-MIPET/PCNet)|

## Reference

If you use this work, please cite:

```bibtex
@article{moglia2025generalist,
  title={Generalist Models in Medical Image Segmentation: A Survey and Performance Comparison with Task-Specific Approaches},
  author={Moglia, Andrea and Leccardi, Matteo and Cavicchioli, Matteo and Maccarini, Alice and Marcon, Marco and Mainardi, Luca and Cerveri, Pietro},
  journal={arXiv preprint arXiv:2506.10825},
  year={2025}
}

```

## License
This work is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/). 

![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)


