gpt_select_first_row_prompt = """You are given two images: 
      (A) the source image to be edited, and 
      (B) a composite image consisting of 4 contiguous horizontal quarters (from top to bottom: quarter 1, quarter 2, quarter 3, quarter 4). 
      Each quarter in (B) shows the top quarter (upper 1/4 crop) of a different full image produced by applying the same editing instruction to the source image. The lower three-quarters of each full image are not shown. Since only the top part is visible, some quarters may show only background without any objects, while others may show objects partially, with the rest continuing into the unseen lower part of the image. 
      
      The editing instruction is: "{}". 
      
      For each of the 4 quarters, make two independent judgments:
      1) instruction_following — does the visible part allow the edited image to plausibly satisfy the instruction?
      2) source_preservation — do the visible regions unrelated to the instruction look reasonably consistent with the source image?
      
      OUTPUT FORMAT (STRICT):
      Return a single-line JSON object with exactly two keys:
      {{"instruction_following": "<q1,q2,q3,q4>", "source_preservation": "<q1,q2,q3,q4>"}}
      - For each key, the value is a lowercase string of exactly four words, each either "pass" or "fail", separated by a comma and a single space, in order: quarter 1, quarter 2, quarter 3, quarter 4.
      - Example: {{"instruction_following": "pass, fail, pass, fail", "source_preservation": "pass, pass, fail, pass"}}
      - No extra keys, punctuation, notes, or explanations.
      
      Focus on whether the editing instruction applied to the source image could plausibly be satisfied given each quarter candidate, and separately whether areas not related to the instruction look reasonably consistent with the source image—keeping in mind that the limited quarter view alone is not evidence of failed preservation.
      
      Guidance for instruction_following:
      - Say "fail" only if the visible part clearly shows that the instruction cannot be satisfied (e.g., added/edited elements are missing or incorrect, or edits strongly contradict the instruction), OR 
      - even if the unseen lower part were completed naturally, the final image would still definitely not satisfy the instruction. If there is any reasonable way the instruction could still be satisfied, say "pass". 
      - Cropping that hides the edit is not a failure by itself.
      
      Guidance for source_preservation:
      - Say "fail" only if the visible part clearly breaks preservation of the source image in areas not required by the instruction (e.g., obvious identity swap, major background replacement unrelated to the instruction, severe artifacts/structural distortions that contradict the source). 
      - The limited quarter view alone is not evidence of failure. Do NOT mark "fail" merely because the visible area looks unchanged; preservation often implies minimal change.
"""

gpt_select_second_row_prompt = """You are given two images:
      (A) the source image to be edited, and
      (B) a composite image consisting of 2 contiguous halves (from top to bottom: half 1, half 2).
      Each half in (B) shows the top half of a different full image produced by applying the same editing instruction to the source image. The bottom halves of those full images are not shown. Some objects may appear only partially (for example, the top half of an object is visible, and the bottom half would appear if the image is completed).

      The editing instruction is: "{}".

      For each of the 2 halves, make two independent judgments:
      1) instruction_following — does the visible part allow the edited image to plausibly satisfy the instruction?
      2) source_preservation — do the visible regions unrelated to the instruction look reasonably consistent with the source image?
      
      OUTPUT FORMAT (STRICT):
      Return a single-line JSON object with exactly two keys:
      {{"instruction_following": "<h1,h2>", "source_preservation": "<h1,h2>"}}
      - For each key, the value is a lowercase string of exactly two words, each either "pass" or "fail", separated by a comma and a single space, in order: half 1, half 2.
      - Example: {{"instruction_following": "pass, fail", "source_preservation": "pass, pass"}}
      - No extra keys, punctuation, notes, or explanations.
      
      Focus on whether the editing instruction applied to the source image could plausibly be satisfied given each quarter candidate, and separately whether areas not related to the instruction look reasonably consistent with the source image—keeping in mind that the limited quarter view alone is not evidence of failed preservation.
      
      Guidance for instruction_following:
      - Say "fail" only if the visible part clearly shows that the instruction cannot be satisfied (e.g., added/edited elements are missing or incorrect, or edits strongly contradict the instruction), OR 
      - even if the unseen lower part were completed naturally, the final image would still definitely not satisfy the instruction. If there is any reasonable way the instruction could still be satisfied, say "pass". 
      - Cropping that hides the edit is not a failure by itself.
      
      Guidance for source_preservation:
      - Say "fail" only if the visible part clearly breaks preservation of the source image in areas not required by the instruction (e.g., obvious identity swap, major background replacement unrelated to the instruction, severe artifacts/structural distortions that contradict the source). 
      - The limited quarter view alone is not evidence of failure. Do NOT mark "fail" merely because the visible area looks unchanged; preservation often implies minimal change.
"""

refine_prompt = """Rewrite the editing instruction "{}" into an extremely concise command that guides the model to finish the edit in the lower half of the image, based on the visual cues from the upper half.

RULES
- BE BRIEF: Use the fewest words possible. Aim for 3-7 words.
- IMPERATIVE: Start with a strong verb (e.g., "Make", "Turn", "Finish", "Remove").
- IGNORE original wording if it's complex; focus only on the visual goal.
- FOCUS ON THE NEXT STEP: Describe what the lower half needs to look like to match the upper half.
- UNCHANGED CASE: If the visible portion suggests the edit is already fully consistent, return the original instruction.

OUTPUT
- Output exactly one short sentence.

EXAMPLES
Original: "Change the animal from a cat to a dog"; Visible: dog head visible →
Output: "Draw the dog's body."

Original: "Change the color of the horse from white to golden"; Visible: golden head visible →
Output: "Make the rest of the horse golden."

Original: "Transform into digital art"; Visible: digital style top →
Output: "Apply digital art style to the bottom."
"""
