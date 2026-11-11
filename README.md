## How to Use `master_pipeline_version_0.3.py`

### 1. Setup & Launch
1. Place matching video and GPS log pairs (e.g., `road1.mp4` and `road1.txt`) inside `data/raw/`.
2. Activate your virtual environment and run the script from the project root:
   ```bash
   python scripts/master_pipeline_version_0.3.py
   ```

### 2. Select a Session
When the GUI window opens to the **Session Selector Menu**:
* **Select a specific pair:** Press the number key (`1`–`9`), use `Up`/`Down` arrows + `Enter`, or click the session row directly with your mouse.
* **Process all pairs sequentially:** Press `A`.
* **Exit:** Press `Q` or `Esc`.

### 3. Annotate Frames
Once synchronized, the pipeline loads 1 frame per second. Use your mouse and keyboard directly inside the window:

| Action | Control |
| :--- | :--- |
| **Draw Bounding Box** | `Left-Click + Drag` directly on the road image |
| **Undo Last Box** | `Right-Click` or `Z` |
| **Mark Frame Clean** | `C` *(saves immediately and loads next frame)* |
| **Save Boxes & Next** | `N` / `Space` / `Enter` |
| **Previous Frame** | `P` or `Left Arrow` |
| **Skip Blurry Frame** | `S` or `Right Arrow` |
| **Return to Session Menu** | `M` |
| **Quit & Save Progress** | `Q` or `Esc` *(auto-resumes on next run)* |

### 4. Assign Damage Class Hotkeys
Immediately after releasing the mouse to draw a box, press one of the following keys to label it:
* **`0`** : `D00` (Longitudinal Crack)
* **`1`** : `D01` (Longitudinal Construction Joint)
* **`2`** : `D10` (Transverse Crack)
* **`3`** : `D11` (Transverse Construction Joint)
* **`4`** : `D20` (Alligator Crack)
* **`5`** : `D40` (Other Corruption / Pothole — *Default on `Enter`/`Space`*)
* **`6`** : `D43` (Crosswalk Blur)
* **`7`** : `D44` (White Line Blur)
* **`8`** : `D50` (Repair D50)
* **`9`** : `Repair` (Repair Surface)
* **`B`** : `Block crack` (Block-shaped cracking)
* **`Esc`** : Cancel current box
