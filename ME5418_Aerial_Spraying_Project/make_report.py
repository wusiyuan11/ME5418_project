"""Generate the verified ME5418 technical report from measured demo outputs."""

# Import CSV parsing so report metrics come from generated data rather than invention.
import csv
# Import mathematical operations for measured root-mean-square error.
import math
# Import filesystem paths for portable input and output locations.
from pathlib import Path
# Import ReportLab colors, page size, styles, units, and document elements.
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Resolve all files relative to this script's project directory.
PROJECT = Path(__file__).resolve().parent
# Select the required final report path.
OUTPUT = PROJECT / "TECHNICAL_REPORT.pdf"
# Select the generated numerical result table.
RESULTS = PROJECT / "outputs" / "simulation_results.csv"

# Read every measured simulation row.
with RESULTS.open("r", encoding="utf-8") as stream:
    # Convert CSV text rows into dictionaries.
    rows = list(csv.DictReader(stream))
# Calculate measured active-waypoint root-mean-square distance.
rmse = math.sqrt(sum(float(row["tracking_error_m"]) ** 2 for row in rows) / len(rows))
# Extract the measured final simulation time.
final_time = float(rows[-1]["time_s"])
# Extract initial and final total mass values.
initial_mass = float(rows[0]["mass_kg"])
final_mass = float(rows[-1]["mass_kg"])
# Reconstruct wind speed from its recorded world-frame vector.
wind_speed = math.hypot(float(rows[0]["wind_x_mps"]), float(rows[0]["wind_y_mps"]))

# Create an A4 document with consistent report margins and metadata.
document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=17 * mm, title="ME5418 Aerial Spraying Technical Report", author="ME5418 Project")
# Start from ReportLab's readable standard style collection.
styles = getSampleStyleSheet()
# Add the project heading style.
styles.add(ParagraphStyle(name="ProjectHeading", parent=styles["Heading1"], textColor=colors.HexColor("#17365D"), fontSize=16, leading=19, spaceBefore=10, spaceAfter=7))
# Add a compact subsection style.
styles.add(ParagraphStyle(name="ProjectSubheading", parent=styles["Heading2"], textColor=colors.HexColor("#2E5E3F"), fontSize=12, leading=15, spaceBefore=8, spaceAfter=5))
# Add a readable body style.
styles.add(ParagraphStyle(name="ProjectBody", parent=styles["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=7))
# Add a small caption style.
styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.3, leading=11, textColor=colors.HexColor("#444444"), alignment=1, spaceAfter=7))
# Create the ordered list of document elements.
story = []


# Append one styled section heading.
def heading(text):
    """Add a numbered section heading; input is text and there is no return value."""
    # Append a heading paragraph using the project style.
    story.append(Paragraph(text, styles["ProjectHeading"]))


# Append one styled subsection heading.
def subheading(text):
    """Add a subsection heading; input is text and there is no return value."""
    # Append a compact subsection paragraph.
    story.append(Paragraph(text, styles["ProjectSubheading"]))


# Append one body paragraph containing safe ReportLab markup.
def body(text):
    """Add explanatory prose; input is markup text and there is no return value."""
    # Append a body paragraph using consistent typography.
    story.append(Paragraph(text, styles["ProjectBody"]))


# Append a project output image at a readable width.
def figure(filename, caption):
    """Add one output image and caption; inputs are filename and caption text."""
    # Add the image while preserving a width that fits the page.
    story.append(Image(str(PROJECT / "outputs" / filename), width=166 * mm, height=100 * mm, kind="proportional"))
    # Add the human-readable figure caption below the image.
    story.append(Paragraph(caption, styles["Caption"]))


# Define a footer callback for every generated page.
def draw_footer(canvas, doc):
    """Draw a report label and page number using ReportLab's page canvas."""
    # Save the current drawing state before changing font and color.
    canvas.saveState()
    # Select a small neutral footer font.
    canvas.setFont("Helvetica", 8)
    # Select a muted footer color.
    canvas.setFillColor(colors.HexColor("#666666"))
    # Draw the project label at the lower-left margin.
    canvas.drawString(18 * mm, 10 * mm, "ME5418 Aerial Spraying Prototype")
    # Draw the page number at the lower-right margin.
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    # Restore the previous drawing state.
    canvas.restoreState()


# Add the report title and scope statement.
story.append(Paragraph("Reinforcement Learning-Based Robust Trajectory Tracking for an Aerial Spraying Quadrotor", styles["Title"]))
story.append(Spacer(1, 5 * mm))
story.append(Paragraph("Validated 2-D Gymnasium Environment and Graphical Demonstration", styles["Heading2"]))
story.append(Spacer(1, 5 * mm))
body("<b>Scope:</b> This report documents the implementation that was actually executed in this package. It is a simplified horizontal point-mass prototype and does not claim full quadrotor aerodynamics, PPO training, or real-flight validation.")
# Add a measured-result summary table.
summary_data = [["Quantity", "Measured seed-7 result"], ["Mission outcome", "Success"], ["Duration", f"{final_time:.1f} s ({len(rows) - 1} steps)"], ["Wind magnitude", f"{wind_speed:.3f} m/s"], ["Total mass", f"{initial_mass:.1f} kg to {final_mass:.1f} kg"], ["Active-waypoint RMSE", f"{rmse:.3f} m"]]
summary_table = Table(summary_data, colWidths=[65 * mm, 95 * mm])
summary_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")])]))
story.append(summary_table)
story.append(PageBreak())

heading("1. Introduction")
body("Aerial spraying requires a vehicle to cover crop rows accurately while its liquid payload decreases and wind pushes the airframe away from the desired route. These disturbances can change the relationship between a controller command and resulting motion. The ME5418 research problem is to establish a reproducible environment in which nominal and robust trajectory tracking can be evaluated before moving to higher-fidelity simulation and learning-based control.")
heading("2. Project objectives")
body("The immediate objective is to validate software structure, path generation, disturbance models, termination logic, reproducibility, and visualization. The longer-term objective is a fair comparison of PID and PPO under nominal, spraying-only, wind-only, combined, and unseen conditions. Advanced RL is intentionally deferred until this base environment is demonstrably correct.")
heading("3. Software architecture")
body("<b>spraying_env.py</b> contains the Gymnasium environment, path generator, PD-style demonstration controller, simulation runner, CSV exporter, static plots, and animation. <b>verify_environment.py</b> checks package imports and invokes Gymnasium's official checker. <b>tests/test_spraying_env.py</b> checks reset, step, spaces, wind, mass, path, completion, error, seeds, and dynamic effects. Documentation explains installation, model scope, and research progression.")
subheading("Libraries used")
body("Gymnasium supplies the standard environment, action space, observation space, reset, and step contracts. NumPy supplies vectors, random sampling, norms, and integration arithmetic. Matplotlib draws figures and animations. Pillow encodes GIF output. Python's csv module saves numerical results without adding Pandas. pytest executes automated verification. ReportLab creates this PDF.")
heading("4. Environment construction")
body("The action is a two-element normalized horizontal acceleration request. The observation is [x, y, vx, vy, ex, ey, route fraction, time fraction, speed]. Mass and wind are available in the diagnostic info dictionary but are hidden from the policy. Reset returns (observation, info); step returns (observation, reward, terminated, truncated, info). The reward encourages target progress and waypoint completion while penalizing target distance, action effort, and field departure.")

heading("5. Trajectory generation")
body("Five spraying lanes are located at y = 2, 6, 10, 14, and 18 m. Lane endpoints alternate between x = 1 and x = 19 m. This creates ten waypoints connected as a continuous lawnmower pattern inside a 20 m square field. The vehicle starts at the first point and targets the second.")
figure("trajectory.png", "Figure 1. Seed-7 reference and actual trajectory. The final state reports mission completion within the 0.38 m waypoint radius.")
story.append(PageBreak())

heading("6. Wind disturbance model")
body("At reset, wind magnitude is sampled as V<sub>w</sub> ~ Uniform(0, 3) m/s and direction as theta<sub>w</sub> ~ Uniform(0, 2*pi). The fixed world-frame vector is v<sub>wind</sub> = [V<sub>w</sub> cos(theta<sub>w</sub>), V<sub>w</sub> sin(theta<sub>w</sub>)]. Relative air velocity is v - v<sub>wind</sub>, and illustrative drag is F<sub>w</sub> = -c ||v - v<sub>wind</sub>|| (v - v<sub>wind</sub>). The coefficient is not experimentally calibrated, wind is constant per episode, and no turbulence or gust spectrum is represented.")
body("Automated tests sampled 30 seeded episodes, verified the [0, 3) m/s range and polar-to-Cartesian equation, and demonstrated that wind changes trajectory. This verifies implementation consistency, not aerodynamic realism.")
heading("7. Time-varying mass model")
body("The implemented equations are m<sub>liquid</sub>(t) = max(m<sub>liquid,0</sub> - q t, 0) and m<sub>total</sub>(t) = m<sub>drone</sub> + m<sub>liquid</sub>(t). Default values are 1.2 kg dry mass, 0.8 kg liquid, and q = 0.02 kg/s. The analytical update prevents cumulative subtraction error. Tests verified monotonic decrease, the dry-mass lower bound, and no decrease after depletion.")
body("Mass affects acceleration through a = a<sub>command</sub> + F<sub>w</sub>/m<sub>total</sub>. A test using the same wind showed larger wind-induced motion at lower mass. The control action itself is defined as acceleration, so mass does not scale that term; this is an explicit simplification.")
figure("mass_curve.png", "Figure 2. Total mass decreases linearly from 2.0 kg to the 1.2 kg dry-mass floor at 40 s and then remains constant.")
story.append(PageBreak())

heading("8. Control method")
body("The current controller is a bounded PD-style waypoint tracker. Position error produces a desired velocity limited component-wise to +/-1.6 m/s. Velocity error is converted to normalized planar acceleration and clipped to the action bounds. This is a deterministic demonstration baseline. It is not a full cascaded position-attitude PID and is not an RL policy.")
heading("9. Visualization")
body("The animated GIF continuously updates the reference path, actual path, drone marker, active target, fixed world-frame wind arrow, wind magnitude/direction, liquid payload, total mass, time, active-waypoint error, and mission state. Static figures capture final path, mass history, and error history. A CSV preserves every recorded numerical sample. MP4 export is implemented but was not produced on the test host because FFmpeg was unavailable.")
figure("tracking_error.png", "Figure 3. Active-waypoint distance. Upward jumps occur when the target advances; this is not continuous-path cross-track error.")
heading("10. Results")
body(f"The executed seed-7 run completed the route in {len(rows) - 1} steps ({final_time:.1f} s). Wind magnitude was {wind_speed:.3f} m/s. Total mass changed from {initial_mass:.1f} kg to {final_mass:.1f} kg. Active-waypoint RMSE was {rmse:.3f} m; its large value reflects long 18 m lane targets and should not be interpreted as cross-track RMSE. The official Gymnasium checker passed, and all 11 automated tests passed.")

heading("11. Limitations")
body("The model is two-dimensional and point-mass only. It omits altitude, attitude, angular rates, rotor thrust, motor dynamics, mixer saturation, ground effect, payload sloshing, changing inertia, sensor noise, actuator delay, and calibrated aerodynamics. Wind is spatially and temporally constant within an episode. The controller observes neither mass nor wind. Therefore, this result establishes software function and reproducibility only; it does not establish real aerial robustness.")
# Start the forward-looking research plan on a fresh page to keep its heading with its text.
story.append(PageBreak())
heading("12. Future improvements")
body("G1 should validate nominal tracking with zero wind and fixed mass. G2 should enable continuous spraying without wind. G3 should randomize wind with fixed mass. G4 should combine both disturbances. G5 should test unseen wind magnitudes, directions, gusts, payloads, and spray rates. Each stage should compare a tuned PID baseline with PPO using identical seeds, success criteria, cross-track RMSE, maximum error, crash rate, control effort, and recovery time. Gain-scheduled PID and recurrent PPO are reasonable later studies after standard PID and feed-forward PPO are stable.")
heading("13. Conclusion")
body("The package now provides an executed and tested Gymnasium-compatible 2-D aerial-spraying prototype, reproducible wind and payload behavior, an improved graphical demonstration, numerical export, automated tests, and transparent documentation. Its simplifications are explicit. The next defensible step is a staged PID/PPO experiment in a validated 3-D quadrotor environment rather than claiming advanced RL performance from this prototype.")

# Build the final PDF and apply the footer to every page.
document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
# Print the generated path for terminal confirmation.
print(OUTPUT)
