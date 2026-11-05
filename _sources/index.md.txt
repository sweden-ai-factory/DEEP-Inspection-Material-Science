# DEEP Inspection for Materials Science


Inspection and characterization of materials are fundamental for understanding material properties, ensuring quality control, and accelerating materials development and advanced manufacturing. However, conventional approaches often require extensive expert analysis of complex images, making them time-consuming and difficult to scale. With the rapid growth of high-resolution imaging techniques, deep learning-based methods provide powerful solutions for automated materials image analysis by learning meaningful visual features for tasks such as defect classification, microstructure recognition, process monitoring, and intelligent quality assessment.



This workshop offers a focused, hands-on introduction to deep learning for visual inspection in materials science. Using the Severstal Steel Defect Detection dataset as a concrete, running example, participants will follow a complete analysis pipeline, from raw image data through model training to interpretable results. The techniques covered are deliberately chosen for their breadth of applicability: while the examples are grounded in industrial quality control, the same approaches translate directly to defect detection in electron microscopy, anomaly identification in medical imaging, and structural characterization across a wide range of domains.


:::{prereq}
- Basic Python programming (loops, functions, libraries)
- Familiarity with NumPy or similar data manipulation tools is helpful but not required
- No prior deep learning experience necessary – core concepts will be introduced from scratch
:::


<!--
:::{prereq}

- FIXME
- XXX
- XXX
:::
-->

<!--
```{csv-table}
:delim: ;
:widths: auto

20 min ; {doc}`filename`
```
-->


```{toctree}
:caption: Setup
:maxdepth: 1

episodes/0-setting-up-programming-environment
episodes/0b-setting-up-YOLO-Week2
```


```{toctree}
:caption: Episodes
:maxdepth: 1

<<<<<<< HEAD
=======
>>>>>>> db201f0a3e8839550ac2b0f51fd9bbd8767f07ca
episodes/1-AnomalyDetection_PatchCore-lesson
episodes/2-CNN-LeNet-AlexNet-lesson
episodes/3-TL-VGG-ResNet-ViT-lesson
episodes/4-YOLO_Unet_SegFormer-lesson
```



<!--
```{toctree}
:caption: Reference
:maxdepth: 1

quick-reference
guide
```
-->
## Learning outcomes

This material is for practitioners and researchers in materials science and industrial inspection
who want to apply deep learning to surface defect analysis. No prior deep learning experience is required,
but familiarity with Python is assumed. The techniques covered transfer directly to related domains
such as electron microscopy, medical imaging, crystallography, and broader manufacturing quality control.

By the end of this workshop, learners should be able to:

- Explain the two core assumptions behind CNNs (translation invariance and locality) and why they
  make convolution more parameter-efficient than fully connected layers
- Describe and distinguish four computer vision paradigms: anomaly detection, image classification,
  object detection, and image segmentation, and identify which is appropriate for a given inspection task
- Apply transfer learning (feature extraction and fine-tuning) with pretrained CNN and transformer
  architectures (VGG, ResNet, ViT) to a new defect classification dataset
- Explain how PatchCore detects anomalies without labeled defect examples, and contrast this with
  supervised classifiers and their open-set failure modes
- Understand the YOLO family's progression from two-stage pipelines to single-pass anchor-free detection,
  and apply YOLO11 to both object detection and instance segmentation tasks
- Distinguish semantic segmentation (U-Net, SegFormer) from instance segmentation (YOLO11-seg) and
  select the appropriate architecture for a given materials inspection scenario

## Credit

This material was developed by the Sweden AI Factory at NAISS/NSC, Linköping University
and RISE.

Parts of the curriculum are adapted from [Dive into Deep Learning](https://d2l.ai/)
(CC BY-SA 4.0).

:::{admonition} Use of AI assistants
:class: seealso dropdown

Parts of this material — including draft text, code cells and example notebooks — were
produced with the help of AI coding assistants (e.g. Anthropic Claude, GitHub Copilot).
All content has been reviewed, tested and edited by the authors, who take full
responsibility for its accuracy.
:::

## License

::::{admonition} Licensing
:class: attention

Unless noted otherwise, this material is **dual-licensed**:

- Teaching material, text, figures and other media —
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Source code, code cells and snippets —
  [MIT](https://opensource.org/license/mit)

Copyright © 2026 Sweden AI Factory and the contributors.

The full license texts are in
[`LICENSE`](https://github.com/MimerMS/DEEP-Inspection-Material-Science/blob/main/LICENSE)
and
[`LICENSE.code`](https://github.com/MimerMS/DEEP-Inspection-Material-Science/blob/main/LICENSE.code)
in the repository.

**Suggested attribution**

> "DEEP Inspection for Materials Science" by Sweden AI Factory, used under CC BY-SA 4.0.

:::{admonition} CC BY-SA 4.0 — what this means
:class: note dropdown

**You are free to**

- **Share** — copy and redistribute the material in any medium or format, for any
  purpose, even commercially.
- **Adapt** — remix, transform and build upon the material, for any purpose, even
  commercially.

The licensor cannot revoke these freedoms as long as you follow the license terms.

**Under the following terms**

- **Attribution** — you must give
  [appropriate credit](https://creativecommons.org/licenses/by-sa/4.0/#ref-appropriate-credit),
  provide a link to the license, and
  [indicate if changes were made](https://creativecommons.org/licenses/by-sa/4.0/#ref-indicate-changes).
- **ShareAlike** — if you remix or build upon the material, you must distribute your
  contributions under the
  [same license](https://creativecommons.org/licenses/by-sa/4.0/#ref-same-license).
- **No additional restrictions** — you may not apply legal terms or
  [technological measures](https://creativecommons.org/licenses/by-sa/4.0/#ref-technological-measures)
  that legally restrict others from doing anything the license permits.

**Notices**

You do not have to comply with the license for elements of the material in the public
domain, or where your use is permitted by an applicable
[exception or limitation](https://creativecommons.org/licenses/by-sa/4.0/deed.en#ref-exception-or-limitation).
No warranties are given; other rights such as
[publicity, privacy or moral rights](https://creativecommons.org/licenses/by-sa/4.0/deed.en#ref-publicity-privacy-or-moral-rights)
may limit how you use the material.

This is a human-readable summary, not a substitute for the
[legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).
:::

:::{admonition} MIT License — full text
:class: note dropdown

Copyright © 2026 Sweden AI Factory and the contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
:::
::::

