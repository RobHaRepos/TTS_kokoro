Third-party Licenses
====================

This project depends on a number of third-party components. The most notable in-repo third-party component is the `kokoro` submodule, which provides the TTS pipeline and models.

1) kokoro (submodule)
   - License: Apache License 2.0
   - License file: `kokoro/LICENSE`
   - Notes: Apache 2.0 requires retaining the license text and may require attribution if a NOTICE file is present. The `kokoro` submodule is included as an external component; any redistribution should include the kokoro license text and maintain notices.

How to comply
-------------
- When distributing compiled or source versions of this project, include this repository's `LICENSE` file and the kokoro submodule's `kokoro/LICENSE` file alongside your distribution.
- If you modify files from `kokoro` and redistribute them, insert a prominent notice that clarifies your changes and preserve the license header in the modified files per the Apache 2.0 rules.
- If the `kokoro` submodule has a `NOTICE` file, include that NOTICE content verbatim in your distribution's `NOTICE` file.

If you include other third-party binary dependencies, list their licenses here and include the license or link to the license in your distribution.

Questions
---------
If you’re unsure whether a library or file requires special attribution (e.g., a patent clause, NOTICE file, or other wording), consult the license text of the third-party project and, when in doubt, consult legal counsel.

