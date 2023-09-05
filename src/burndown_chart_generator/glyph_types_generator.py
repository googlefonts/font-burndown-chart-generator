# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

from ufoLib2 import Font


def print_glyph_types_for(ufo_path: Path):
    font = Font.open(ufo_path)
    print("[glyph_types]")
    for glyph in font:
        assert glyph.name
        name = glyph.name if not "." in glyph.name else f'"{glyph.name}"'
        glyph_type = "drawn" if glyph.contours else "composite"
        print(f'{name} = "{glyph_type}"')
