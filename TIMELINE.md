# Drawing Assistant Timeline

Project tracker for the Augmented Vision assignment.

Use this file as a lightweight shared board:
- Change `[ ]` to `[x]` when a task is done.
- Move tasks between sections as the project progresses.
- Keep visual validation in mind for every major feature.

> Note: This is a visual project, so each important step should be tested by displaying the result on real input images or video frames.

## Key Dates

- [ ] First interim report — 15.06.2026
- [ ] Second interim report + final presentation video — 13.07.2026
- [ ] Final presentation — TBA

## Milestones

### Milestone 1: Detect the Paper Surface 

- [ ] 1.1 Make paper detection work reliably
- [ ] 1.2 Add scaling and adaptation of the paper as input for drawing detection

### Milestone 2: Detect the Drawing

- [ ] 2.1 Extract contours and filter out noise
- [ ] 2.2 Group detected lines into meaningful segments
- [ ] 2.3 Output a list of detected line segments, including coordinates, angles, and lengths

### Extra Ideas

#### Paper Pattern

- [ ] Detect the paper pattern
- [ ] Build a pipeline to remove the pattern and obtain a clean paper representation

#### Coloring

- [ ] Extend the system to support different colors, not only contours

## Current Sprint Board

### To Do (Not assigned)

- [ ] Test detection on multiple example images
- [ ] Prepare content for the first interim report

### In Progress

#### Olha

- [ ] Improve paper edge detection stability

#### Lara

- [ ] Support both local image input and video URL input


### Done

- [x] Set up the basic project structure — Lara
- [x] Add the initial paper line detection logic — Lara
- [x] Refactor the code into smaller modular functions — Olha
- [x] Document the functions in `project.py` — Olha

## Notes

- Keep tasks short and action-oriented.
- Add owners when a task is assigned.
- Add new tasks as soon as new requirements appear.
- Maintain a modular code structure and use Google-style docstrings for documentation. 
