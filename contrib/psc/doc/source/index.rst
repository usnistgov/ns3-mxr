
:orphan: true

.. only:: html or latex

.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ============= Overall title
   ############# Chapter
   +++++++++++++ Section
   ============= Subsection (#.#.#) (or -----)
   ************* Paragraph

ns-3 PSC Video Streaming documentation
======================================

This document provides information for the standalone version of the video streaming feature from the ns-3 **PSC module**. The PSC module itself is a comprehensive suite available for ns-3 simulations.

For users interested specifically in video streaming functionalities without the need for the entire PSC module, this standalone version has been prepared.

Source Repository
-----------------

The full PSC module can be found on GitHub at the following URL:

- `PSC Module for ns-3 <https://github.com/usnistgov/psc-ns3/tree/psc-7.0/src/psc>`_

This standalone version is a trimmed-down iteration of the original module, focusing solely on video streaming features. It has been created to facilitate easier integration and usage within ns-3 projects that require only video streaming capabilities.
Additionally, it includes within the `examples/mxr_cdf` directory the distribution files for **Reduced Function Headset (RFH) Medical Extended Reality (MXR) applications**. 

For further details on the applications and their usage in this context, please refer to the following article:

- `RFH MXR Applications in ns-3 <http://example.com/article>`_


.. toctree::
   :maxdepth: 2

   video-models

