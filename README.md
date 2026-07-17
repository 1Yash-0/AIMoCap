# aimocap — Browser-Based AI Motion Capture

## What Is This Project?

**aimocap** is a motion capture tool that runs entirely inside a web browser. A user records a performance from multiple smartphone cameras, drops the videos into our web app, and gets back a 3D character animation they can use directly in Unreal Engine, Unity, Blender, or any game engine.

**The entire pipeline runs locally on the user's machine.** No video is uploaded to any server. We use **WebGPU** — a new browser technology that lets us run AI models directly on the user's graphics card, right from the browser tab.

No servers to pay for. No privacy concerns. No downloads or installations.

---

## Who Is This For?

Our target users are **indie game developers and solo animators** who can't afford professional motion capture studios ($500–$5,000+ per session). With aimocap, all they need is:
- 3 smartphones (any modern phone)
- A living room with enough space to move
- A computer with a decent graphics card
- A web browser (Chrome 113+ or Edge 113+)

---

## The Full User Journey

Here is exactly what happens from the user's perspective, step by step:

### Step 1: Record the Performance

The user places 3 or more smartphones around the room, angled towards the center where they'll perform. The phones don't need to be fancy — any modern phone with a camera works.

They press record on all the phones (doesn't have to be perfectly synchronized — we handle that), step into the center, and **clap their hands once**. That clap creates a sharp sound spike in every video's audio. Our software uses this to align all the videos in time later.

Then they perform their action — walking, dancing, fighting, whatever they need for their game. When done, they stop recording and transfer the videos to their computer.

### Step 2: Open the Web App

The user navigates to our web app in Chrome. No login needed for the free tier. They see a clean, modern interface.

### Step 3: Upload Videos + Character Model

The user drags and drops:
- Their **3+ video files** (MP4 or WebM from their phones)
- Their **character model file** (`.fbx` — like Unreal's Manny mannequin, or their own custom game character)

Everything stays on their machine. The files are read directly by the browser; nothing is sent over the internet.

### Step 4: GPU Check & Model Selection

When the app first loads (or on first use), it runs a quick invisible test on the user's graphics card to figure out how powerful it is. Based on the result, it recommends the best AI model:

| GPU Power | Example Hardware | Recommended Model | Quality |
|-----------|-----------------|-------------------|---------|
| Low | Intel integrated graphics, older laptops | RTMPose-M | Good body tracking, weaker hands |
| Medium | GTX 1060, RX 580, M1 Pro | RTMPose-L | Good body + hand tracking |
| High | RTX 3060, RX 6700, M2 Pro | ViTPose-B | Great full body |
| Very High | RTX 4060+, RX 7800+, M3 Max | ViTPose-H | Best quality available |

The user sees the recommendation and can override it if they want. There's also an estimated processing time shown (e.g., "~8 minutes for 1 minute of footage").

They can also click a **"Run test frame"** button that processes a single frame and shows the AI's keypoint overlay on top of the video — so they can visually compare quality between models before committing.

### Step 5: Processing (This Is Where the Magic Happens)

The user clicks "Process" and the pipeline begins. This is all happening locally in the browser. We need a **beautiful, informative processing screen** here — this could take several minutes depending on the video length and GPU power.

Here's what happens behind the scenes in order:

#### 5a. Audio Sync (a few seconds)
The app extracts the audio from each video, applies a bandpass filter to isolate the clap's frequency range (2–8 kHz), and uses FFT cross-correlation to find the exact moment of the clap in each video. This gives us millisecond-accurate time alignment between all the cameras.

**UI moment:** Show "Syncing camera timelines..." with a quick check animation. This is fast.

#### 5b. 2D Pose Estimation — the heavy WebGPU step (the bulk of processing time)
This is the most time-consuming step and where WebGPU really matters. For every single frame of every single video, the AI:
1. Finds the person in the frame (using YOLOX-Nano, a tiny person detector)
2. Crops around them
3. Runs the pose model to find **133 keypoints** on their body

Those 133 keypoints cover:
- **17 body joints** — nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **6 foot points** — big toe, small toe, heel (both feet)
- **68 face landmarks** — full face contour (jawline, eyebrows, nose bridge, lips, eye corners)
- **21 left hand joints** — wrist + 4 joints per finger (thumb, index, middle, ring, pinky)
- **21 right hand joints** — same as left, mirrored

Each keypoint comes with a confidence score (0.0 to 1.0) that tells us how sure the AI is about that point. This confidence is critical for later steps.

**How it runs on the GPU:** We use ONNX Runtime Web with the WebGPU execution provider. The AI model (an `.onnx` file) is downloaded once from our CDN and cached in the browser (via IndexedDB or Cache API) so it doesn't need to re-download. The largest model (ViTPose-H at FP16 precision) is about 1.2 GB — we show a download progress bar on first use.

We run each camera's inference in a separate **Web Worker** (a background thread in the browser) so the UI stays responsive and doesn't freeze. On a 3-camera setup, that's 3 workers sharing the GPU in parallel.

**UI moment:** This is the long step. Show per-camera progress bars (e.g., "Camera 1: 450/1800 frames"), overall progress, estimated time remaining, and maybe a live preview of keypoints being detected on sample frames to keep the user engaged.

#### 5c. Camera Calibration (a few seconds)
The app needs to figure out where each camera was physically positioned in the room and which direction it was pointing. We do this automatically from the 2D poses — the AI knows where the person's joints appear in each camera view, and from the geometry of how those views differ, we can compute the camera positions.

For v1, we may ask the user to do a brief calibration routine first — stand at 4-5 known positions (corners of a square marked with tape on the floor) for 2 seconds each. This gives us very reliable calibration.

**UI moment:** If using guided calibration, show a friendly instruction screen (possibly with a simple animation showing what to do). If using auto-calibration, show "Calculating camera positions..." with a brief animation.

#### 5d. 3D Triangulation (near-instant)
Now we have 2D keypoints from each camera plus the camera positions. The app combines these to compute the true 3D position of every joint in every frame.

The math: if Camera A sees the left elbow at pixel (400, 300) and Camera B sees it at pixel (200, 500), and we know where both cameras are, we can draw invisible lines from each camera through those pixels and find where they intersect in 3D space. That intersection is the real-world position of the elbow.

We weight each camera's contribution by its confidence score. If a camera can barely see the hand (low confidence), it gets less influence than a camera with a clear view.

**UI moment:** Very fast, maybe just a quick "Building 3D skeleton..." message.

#### 5e. Temporal Smoothing (near-instant)
Raw 3D positions have small frame-to-frame jitters. We smooth them out:
- **One-Euro filter** — an adaptive filter that smooths slow movements but stays responsive during fast movements (no lag)
- **Bone-length stabilization** — real bones don't change length, so we enforce consistent bone lengths across all frames
- **Optional smoothness slider** — the user can control how smooth vs. raw the result is (more smoothing = more cinematic, less smoothing = more snappy/responsive for gameplay)

**UI moment:** Could show a before/after jitter comparison, or just a quick "Smoothing animation..." step.

#### 5f. Skeleton Retargeting (a few seconds)
This is the **killer feature** that sets us apart. We take the generic 3D skeleton and map it onto the user's specific character model.

The challenge: the captured human skeleton and the game character skeleton have different proportions, different bone names, different coordinate systems, and different rest poses. A real human's arm might be 60cm, but the game character's arm might be 30cm (or 200cm for a monster). We need to transfer the *motion* (the rotations), not the positions.

**How it works:**
1. **Parse the user's FBX** — we read their character's skeleton hierarchy (which bones connect to which, their rest pose)
2. **Auto-map bones** — we fuzzy-match our keypoint names to their bone names (e.g., "left_elbow" → "LeftLowerArm" or "Bip01_L_Forearm" or "mixamorig:LeftForeArm"). Most standard rigs auto-map completely.
3. **Manual mapping fallback** — if some bones can't be auto-matched (custom rig with unusual names), we show a drag-and-drop mapping UI
4. **Transfer rotations** — for each frame, for each bone, we compute the rotation that makes the character's bone point in the same direction as the captured motion, adjusted for the character's own proportions
5. **Foot locking** — we detect when feet are on the ground and lock them in place so there's no "ice skating" (foot sliding)

**UI moment:** This is a great place for a 3D preview — show the user's character model in a viewport with the captured animation playing on it. Let them scrub through the timeline.

#### 5g. Export
The animation is packaged into a downloadable file:
- **BVH** — a plain-text industry-standard animation format. Works everywhere (Unreal, Unity, Blender, Maya, MotionBuilder). Always available.
- **FBX** — the binary format Unreal Engine prefers. 
- **GLB/GLTF** — modern format used by Godot, Bevy, Babylon.js, and others.

**UI moment:** Big "Download" button. Maybe offer a dropdown for format selection. Show file size.

### Step 6: Done!

The user downloads their animation file, imports it into their game engine, and it just works. The whole process took a few minutes and never left their computer.

---

## The Tech Stack (What We're Building With)

| Layer | Technology | What It Does |
|-------|-----------|-------------|
| **Build System** | Vite + TypeScript | Fast development, type safety, modern JS tooling |
| **AI Runtime** | ONNX Runtime Web (WebGPU backend) | Runs pose estimation AI models on the user's GPU |
| **Pose Models** | CIGPose-X / RTMPose / ViTPose (ONNX format) | Detects 133 body/face/hand keypoints per frame |
| **Person Detector** | YOLOX-Nano (ONNX) | Finds people in each frame before pose estimation |
| **3D Engine** | Three.js | 3D skeleton visualization, FBX/GLB loading, retargeting preview, animation export |
| **Video Decoding** | WebCodecs API (VideoDecoder) | Hardware-accelerated frame extraction from video files |
| **Audio Processing** | Web Audio API (OfflineAudioContext) | Extract and analyze audio for clap-based synchronization |
| **Math** | gl-matrix | Linear algebra for triangulation, camera calibration |
| **Background Processing** | Web Workers | Run AI inference in background threads so the UI doesn't freeze |
| **Model Storage** | IndexedDB / Cache API | Cache downloaded AI models so they don't re-download |
| **Animation Export** | Custom BVH writer + Three.js GLTFExporter | Generate downloadable animation files |
| **Hosting** | Static hosting (Vercel / Cloudflare Pages) + CDN | Near-zero server costs since all processing is client-side |

---

## Why WebGPU Is the Core Technology

### What Is WebGPU?
WebGPU is a modern browser API that gives web applications direct access to the computer's graphics card (GPU). Before WebGPU, browsers could only use WebGL — which was designed for drawing 3D graphics, not for running AI computations. WebGPU is designed for both, and it's almost as fast as running native code.

### Why It Matters for Us

**1. Zero Server Costs**
Running AI models on cloud GPUs costs $0.50–$2.00 per minute of video processed. For a tool meant for indie devs (who are often broke), that's not sustainable. If we processed 1,000 users per day each doing 2 minutes of footage, we'd be paying $1,000–$4,000/day in GPU costs.

With WebGPU, the user's own computer does all the work. Our cost per user is essentially zero (just CDN bandwidth for the initial model download, which is ~1.2 GB and gets cached).

**2. Total Privacy**
Motion capture videos show real people performing in their homes, studios, or offices. Nobody wants their personal video uploaded to someone else's server. With WebGPU, the video files never leave the user's machine — the browser reads them directly from disk, processes them locally, and outputs the animation file locally.

**3. No Installation Required**
Alternative tools (like OpenPose, MediaPipe, or professional mocap suites) require installing Python, CUDA drivers, specific library versions, etc. Our tool requires... opening Chrome. If their browser supports WebGPU (Chrome 113+, Edge 113+, Firefox coming soon), it just works.

### The Hardware Diversity Challenge
Unlike a server where we control the hardware, WebGPU runs on everything from a $200 Chromebook to a $3,000 gaming rig. The GPU capabilities vary wildly. That's why we have the **GPU benchmark system** — we silently test the user's GPU performance at startup and automatically pick the right AI model tier.

WebGPU also intentionally hides exact hardware specs for privacy (you can't just ask "what GPU does this user have?"). So we measure performance empirically: we run a small matrix multiplication on the GPU, time it, and classify the result into tiers.

### Browser Support
- ✅ Chrome 113+ (stable since May 2023)
- ✅ Edge 113+ (same Chromium engine)
- ⏳ Firefox (in development, behind a flag)
- ⏳ Safari (WebGPU available on macOS Sonoma+, limited)
- ❌ Mobile browsers (not yet, but coming)

If WebGPU isn't available, we fall back to **WASM CPU mode** — much slower, but still works. We'd recommend the lightest model (RTMPose-S) and warn the user about processing times.

---

## Business Model: Free vs. Pro

Since processing is free for us (it runs on the user's GPU), we can offer a generous free tier:

| Feature | Free | Pro |
|---------|------|-----|
| Number of cameras | 3 | 3–12 |
| Max clip length | 2 minutes | Unlimited |
| Export formats | BVH only | BVH + FBX + GLB |
| Character rigs | 1 per session | Unlimited |
| Face tracking | Basic (23 points) | Full (68 points) |
| Finger tracking | ✅ Yes | ✅ Yes |
| Batch processing | 1 clip at a time | Queue multiple clips |
| Our server cost per user | ~$0 | ~$0 |

The key insight: **free users cost us almost nothing** (just CDN bandwidth for the initial model download). Pro revenue is nearly pure margin.

---

## Competitive Advantage

| What Competitors Do | What We Do | Why Ours Wins |
|--------------------|-----------|---------------|
| Cloud GPU inference ($0.50–$2/min per user) | Client-side WebGPU | **$0 cost per user** — sustainable free tier |
| Upload raw video to their servers | All processing in-browser | **Total privacy** — nothing leaves the device |
| Single-camera depth estimation (guessing depth) | Multi-camera triangulation (real 3D math) | **True 3D** — no depth guessing, actual geometry |
| Generic skeleton output that needs manual cleanup | Retarget onto user's exact character | **Zero cleanup** — animation fits their character directly |
| Desktop app installation (Python, CUDA, drivers) | Open a browser tab | **Zero friction** — works on any WebGPU browser |

---

## UI/UX Design Goals

These are the design principles we want the interface to follow:

### 1. Premium & High-Tech Aesthetic
The app should feel like professional creative software, not a student project. Think:
- Dark mode by default (most creative professionals prefer dark interfaces)
- Glassmorphism elements (frosted glass effect on panels)
- Subtle gradient accents (not flat primary colors)
- Micro-animations on interactions (buttons, transitions, hover states)
- Modern typography (Inter, Outfit, or similar)
- Clean, spacious layouts with clear visual hierarchy

### 2. Trust & Privacy Messaging
Users will be skeptical about uploading videos to a web app. We need to **visually hammer home** that everything is local:
- Show a persistent "Local Processing" badge/icon in the header
- During processing, maybe show a "Your files never leave this device" message
- Use a lock icon or shield iconography
- No login required for basic usage (reinforces that we're not tracking them)

### 3. The Processing Screen Is the Product
Users will spend the most time staring at the processing screen. This cannot be a boring spinner. Ideas:
- Show the current pipeline stage with a description of what's happening
- Per-camera progress bars with frame counts
- Overall estimated time remaining
- Live preview: show sample frames with keypoints drawn on top as they're being detected
- A 3D skeleton preview that builds up as frames are processed
- Fun stats: "Detected 24,700 keypoints so far..."

### 4. The 3D Preview Is the Wow Moment
After processing, show the user's own character model performing the captured motion in a 3D viewport. This is the moment they go "holy crap, it actually works." Make this viewport:
- Interactive (orbit, pan, zoom with mouse/touch)
- Have playback controls (play, pause, scrub, speed control)
- Show a timeline with the animation
- Let them toggle between the raw 3D skeleton and the retargeted character
- Have a ground plane / grid for spatial reference

### 5. Drag-and-Drop Is the Primary Interaction
The very first thing a user does is drag files into the browser. This has to feel satisfying:
- Large drop zone with clear visual feedback (glow, pulse, color change on drag-over)
- Show file thumbnails after drop (video thumbnails, FBX icon)
- Auto-detect file types (video vs. FBX)
- Validate files immediately (wrong format? too short? no audio track? — tell them right away)

### 6. Adaptive UI Based on GPU Tier
The interface should subtly adapt to the user's hardware:
- Low-end GPU → simpler 3D preview (fewer shader effects), recommend lighter models
- High-end GPU → richer 3D preview, more visual effects in the UI itself
- Show the GPU benchmark result in a friendly way ("Your GPU: Great! ⚡" not "4.7 TFLOPS")

---

## Key Pages / Screens

Here's a rough outline of the screens we need:

### 1. Landing Page
- Hero section explaining the product in one sentence
- "Try it free" call to action
- 3-step visual explainer (Record → Upload → Animate)
- Video demo of the full workflow
- Pricing comparison table

### 2. Upload / Drop Zone
- Large drag-and-drop area
- File list after upload (thumbnails, file sizes, durations)
- FBX upload slot (separate or integrated)
- "Process" button

### 3. GPU Benchmark & Model Selection
- Could be a modal or inline panel
- Show recommended model with estimated time
- Dropdown to override
- "Run test frame" button with before/after comparison

### 4. Processing Dashboard
- Pipeline stage indicator (which step we're on, out of 7)
- Per-camera progress
- Live preview area
- Time remaining
- Cancel button

### 5. Preview & Export
- 3D viewport with the animated character
- Playback controls
- Export format selector (BVH / FBX / GLB)
- Download button
- Optional: side-by-side with original video for comparison

### 6. Bone Mapping UI (if auto-map fails)
- Split view: our skeleton on the left, their skeleton on the right
- Drag lines between matching bones
- Color-coded (green = matched, red = unmatched)
- "Auto-detect" retry button

---

## Development Phases

### Phase 1: Foundation (Weeks 1–4)
Get a single camera working end-to-end: upload a video → see 2D keypoints overlaid on the frames.
- Project scaffolding (Vite + TypeScript + Three.js)
- WebGPU detection and GPU benchmark
- ONNX Runtime Web integration
- Video frame extraction (WebCodecs API)
- Basic upload → keypoint visualization

### Phase 2: Multi-Camera Core (Weeks 5–8)
Add synchronization, calibration, and 3D reconstruction.
- Audio sync (clap detection + cross-correlation)
- Multi-video upload UI
- Camera calibration
- 3D triangulation
- 3D skeleton viewer (Three.js)

### Phase 3: Quality Pipeline (Weeks 9–12)
Temporal smoothing, model tier system, processing UI.
- One-Euro filter + bone stabilization
- Multi-model support and GPU tier recommendation
- Progress tracking UI with live preview

### Phase 4: Retargeting & Export (Weeks 13–16)
The killer feature: character import + animation export.
- FBX/GLB skeleton parsing
- Auto bone mapping + manual mapping UI
- Retargeting math
- BVH/FBX/GLB export
- 3D preview with retargeted character

### Phase 5: Polish & Launch (Weeks 17–20)
Production quality, error handling, landing page.
- Model caching in browser
- Error recovery
- Landing page and docs
- Performance optimization

---

## Glossary

| Term | What It Means |
|------|--------------|
| **Keypoint** | A specific point on the body that the AI tracks — like "left elbow" or "right thumb tip" |
| **Pose estimation** | Using AI to find keypoints in a 2D image/video |
| **WebGPU** | A browser technology that lets web apps run heavy computations on the computer's graphics card |
| **ONNX** | A portable file format for AI models — we can train a model in Python and run it in the browser |
| **ONNX Runtime Web** | The JavaScript library that actually runs ONNX models, with a WebGPU backend for GPU acceleration |
| **Web Worker** | A background thread in the browser — lets us run AI inference without freezing the UI |
| **WebCodecs** | A browser API for efficiently extracting individual frames from video files |
| **Triangulation** | Combining 2D observations from multiple camera angles to calculate true 3D positions |
| **DLT** | Direct Linear Transform — the specific math algorithm for triangulation |
| **IK (Inverse Kinematics)** | Math that figures out joint angles to reach a target position (e.g., "how should the elbow bend so the hand reaches here?") |
| **FK (Forward Kinematics)** | The opposite — given joint angles, compute where hands/feet end up |
| **Retargeting** | Taking captured motion and applying it to a different character with different proportions |
| **Rest pose** | The default stance a character is modeled in (usually T-pose or A-pose) before any animation |
| **BVH** | Biovision Hierarchy — a plain-text animation file format. Stores skeleton structure + per-frame rotations |
| **FBX** | Filmbox — a binary animation/model format used by Unreal Engine, Maya, and most game studios |
| **GLB / GLTF** | A modern, open-source 3D format used by Godot, Babylon.js, and the web 3D ecosystem |
| **Three.js** | The JavaScript library for 3D graphics in the browser — we use it for previewing and exporting |
| **Confidence score** | A number (0.0 – 1.0) the AI gives each keypoint — high = sure, low = guessing |
| **One-Euro filter** | A smoothing algorithm that adapts: smooth during slow movements, responsive during fast movements |
| **Foot locking** | Detecting when feet touch the ground and locking them in place to prevent "ice skating" artifacts |
| **Bundle adjustment** | A math technique that refines camera positions by minimizing reprojection errors across all views |
| **CDN** | Content Delivery Network — a global network of servers for fast file downloads (we host AI models here) |
| **IndexedDB** | A browser database for storing large files locally (we use it to cache downloaded AI models) |
| **FP16 / FP32** | Number precision formats. FP16 (half precision) uses half the memory and runs faster, with minimal quality loss for AI |
