#%% # import libraries
import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cairosvg

from cartopy.io import shapereader
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.legend_handler import HandlerBase
from pathlib import Path

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7
})

#%% # Automatically identify the project root directory

current_dir = Path(__file__).resolve().parent

PROJECT_ROOT = next(
    (
        parent for parent in [current_dir, *current_dir.parents]
        if all(
            (parent / folder).is_dir()
            for folder in ["raw_data", "model", "visualization"]
        )
    ),
    None
)

if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root directory not found.")

path = PROJECT_ROOT / "visualization" / "fig01"
path_fig = path

#%% # fig01 - plot for JJA

init_month = 5

# Load consolidated station metadata
df_metadata = pd.read_csv(path / "stations_train_test_metadata_fig01.csv")

#%% # selected stations

stations_target = [
    2120,
    2153,
    46,
    189,
    568,
    2288
]

# Retrieve target station coordinates from metadata
metadata_station_ids = pd.to_numeric(
    df_metadata["station_id"],
    errors="coerce"
)

df_target = df_metadata.loc[
    metadata_station_ids.isin(stations_target),
    ["station_id", "lon", "lat"]
].copy()

df_target["station_id"] = pd.to_numeric(
    df_target["station_id"]
).astype(int)

# Check target station uniqueness
if df_target["station_id"].duplicated().any():
    raise ValueError("Duplicate target station IDs found in metadata.")

# Check whether all target stations were found
missing_stations = set(stations_target) - set(df_target["station_id"])

if missing_stations:
    raise ValueError(
        f"Target stations not found in metadata: {missing_stations}"
    )

# Preserve the specified station order
df_target = (
    df_target
    .set_index("station_id")
    .loc[stations_target]
    .reset_index()
)

# Check target station coordinates
if df_target[["lon", "lat"]].isna().any().any():
    raise ValueError("Missing coordinates for target stations.")


#%% # select training and assessment stations

# Define seasonal columns
used_column = f"used_s{init_month}"
samples_column = f"n_samples_s{init_month}"
years_column = f"n_years_s{init_month}"

# Convert seasonal usage information to boolean
df_metadata[used_column] = (
    df_metadata[used_column]
    .astype("string")
    .str.lower()
    .map({"true": True, "false": False, "1": True, "0": False})
    .fillna(False)
)

# Select training stations
df_train_plot = df_metadata[
    (df_metadata["data_type"] == "train") &
    (df_metadata[used_column])
].copy()

# Select assessment stations
df_test_used_plot = df_metadata[
    (df_metadata["data_type"] == "test") &
    (df_metadata[used_column]) &
    (df_metadata[samples_column] >= 45)
].copy()

# Preserve the original plotting variable names
df_train_plot["n_years_at_least_one_obs"] = df_train_plot[years_column]
df_test_used_plot["n_years_at_least_one_obs"] = df_test_used_plot[years_column]

# Keep only one row per station for mapping
df_train_plot = df_train_plot.drop_duplicates(
    subset="station_id"
).copy()

df_test_used_plot = df_test_used_plot.drop_duplicates(
    subset="station_id"
).copy()

# Check selected stations
print("Stations in metadata:", df_metadata["station_id"].nunique())
print("Training stations:", df_train_plot["station_id"].nunique())
print("Assessment stations:", df_test_used_plot["station_id"].nunique())
print("Training points to plot:", len(df_train_plot))
print("Assessment points to plot:", len(df_test_used_plot))
print("Selected stations to highlight:", len(df_target))


#%% # load SVG marker

pin_svg_path = path / "station_map_pin.svg"

# Read SVG dimensions
svg_root = ET.parse(pin_svg_path).getroot()

viewbox = svg_root.attrib.get("viewBox")

if viewbox is not None:
    _, _, svg_width, svg_height = map(float, viewbox.split())
else:
    svg_width = float(
        svg_root.attrib["width"].replace("px", "")
    )
    svg_height = float(
        svg_root.attrib["height"].replace("px", "")
    )

# Convert SVG to a high-resolution transparent PNG
pin_output_width = 1024
pin_output_height = int(
    pin_output_width * svg_height / svg_width
)

pin_png = cairosvg.svg2png(
    url=str(pin_svg_path),
    output_width=pin_output_width,
    output_height=pin_output_height
)

pin_image = mpimg.imread(
    io.BytesIO(pin_png),
    format="png"
)

# Check transparency
if pin_image.ndim != 3 or pin_image.shape[2] != 4:
    raise ValueError(
        "The SVG marker must have an RGBA transparency channel."
    )

# Detect the visible marker boundaries
alpha = pin_image[:, :, 3]

visible_rows, visible_cols = np.where(alpha > 0.01)

if len(visible_rows) == 0:
    raise ValueError("The SVG marker is completely transparent.")

ymin = visible_rows.min()
ymax = visible_rows.max()
xmin = visible_cols.min()
xmax = visible_cols.max()

# Crop transparent margins
pin_image = pin_image[
    ymin:ymax + 1,
    xmin:xmax + 1
].copy()

# Automatically identify the bottom tip
bottom_mask = np.where(
    pin_image[:, :, 3][-1, :] > 0.01
)[0]

if len(bottom_mask) == 0:
    raise ValueError("Could not identify the marker tip.")

tip_x = bottom_mask.mean()

pin_anchor = (
    tip_x / pin_image.shape[1],
    0.0
)

print("SVG marker loaded:", pin_image.shape)
print("Marker anchor:", pin_anchor)


#%% # figure

fig = plt.figure(figsize=(6.3, 4.2))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [-180, 180, -56, 75],
    crs=ccrs.PlateCarree()
)


# ===============================
# Map background
# ===============================

ax.add_feature(
    cfeature.LAND,
    facecolor="white",
    edgecolor="black",
    linewidth=0.5
)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=0.35
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.25
)


# ===============================
# Rivers
# ===============================

rivers_shp = (
    path / "shp" / "ne_10m_rivers_lake_centerlines"
    / "ne_10m_rivers_lake_centerlines.shp"
)

reader = shapereader.Reader(str(rivers_shp))

ax.add_geometries(
    reader.geometries(),
    ccrs.PlateCarree(),
    facecolor="none",
    edgecolor="steelblue",
    linewidth=0.15,
    alpha=0.3,
    zorder=1
)

gl = ax.gridlines(
    draw_labels=False,
    linewidth=0.1,
    linestyle="--"
)

gl.xlocator = mticker.FixedLocator(
    [-180, -120, -60, 0, 60, 120, 180]
)

gl.ylocator = mticker.FixedLocator(
    [-30, 0, 30, 60]
)


# ===============================
# Ticks
# ===============================

xticks = [-180, -120, -60, 0, 60, 120, 180]
yticks = [-30, 0, 30, 60]

ax.set_xticks(
    xticks,
    crs=ccrs.PlateCarree()
)

ax.set_yticks(
    yticks,
    crs=ccrs.PlateCarree()
)


def fmt_lon(x):
    return f"{abs(int(x))}° W" if x < 0 else f"{int(x)}° E" if x > 0 else "0°"


def fmt_lat(y):
    return f"{abs(int(y))}° S" if y < 0 else f"{int(y)}° N" if y > 0 else "0°"


ax.set_xticklabels(
    [fmt_lon(x) for x in xticks]
)

ax.set_yticklabels(
    [fmt_lat(y) for y in yticks]
)

ax.tick_params(
    axis="both",
    direction="out",
    length=4,
    width=1,
    top=True,
    right=True,
    labeltop=False,
    labelright=False
)


# ===============================
# Colormap
# ===============================

bounds = [1, 10, 15, 20, 25]

cmap = LinearSegmentedColormap.from_list(
    "red_to_green",
    ["#d73027", "#fee08b", "#006400"]
)

norm = BoundaryNorm(
    bounds,
    cmap.N
)


# ===============================
# Plot training stations
# ===============================

ax.scatter(
    df_train_plot["lon"],
    df_train_plot["lat"],
    c=df_train_plot["n_years_at_least_one_obs"],
    cmap=cmap,
    norm=norm,
    s=1.5,
    marker="^",
    edgecolors="none",
    linewidths=0.2,
    transform=ccrs.PlateCarree(),
    zorder=4
)


# ===============================
# Plot assessment stations
# ===============================

colors_test = cmap(
    norm(df_test_used_plot["n_years_at_least_one_obs"])
)

ax.scatter(
    df_test_used_plot["lon"],
    df_test_used_plot["lat"],
    facecolors="none",
    edgecolors=colors_test,
    s=2,
    marker="o",
    linewidths=0.2,
    transform=ccrs.PlateCarree(),
    zorder=5
)


# ===============================
# Plot selected stations using SVG marker
# ===============================

# Marker size in display points
pin_zoom = 0.010

for _, row in df_target.iterrows():

    pin = AnnotationBbox(
        OffsetImage(
            pin_image,
            zoom=pin_zoom,
            interpolation="lanczos"
        ),
        (row["lon"], row["lat"]),
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
        box_alignment=pin_anchor,
        frameon=False,
        pad=0,
        annotation_clip=True,
        zorder=8
    )

    ax.add_artist(pin)


# consolidated legend

# ===============================
# Train / Assessment / Hydrography
# ===============================

legend_markers = [
    Line2D(
        [0], [0],
        marker="^",
        linestyle="None",
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.6,
        markersize=3.5,
        label="Train"
    ),

    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=3.5,
        markeredgewidth=0.6,
        label="Assessment"
    ),

    Line2D(
        [0], [0],
        color="steelblue",
        linewidth=0.7,
        label="Hydrography"
    )
]


# ===============================
# Data length classes
# ===============================

from matplotlib.text import Text


class SectionHeader(Line2D):
    def __init__(self):
        super().__init__([], [], label="")


class HandlerSectionHeader(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans
    ):
        return [
            Text(
                x=-xdescent,
                y=-ydescent + height / 2,
                text="Data length (years)",
                fontsize=fontsize,
                ha="left",
                va="center",
                transform=trans,
                clip_on=False
            )
        ]


section_handle = SectionHeader()


colors_bins = [
    "#d73027",
    "#fdae61",
    "#a6d96a",
    "#1a9850"
]

labels_bins = [
    "1–10",
    "10–15",
    "15–20",
    "20–25"
]

legend_bins = [
    Patch(
        facecolor=c,
        edgecolor="none",
        label=l
    )
    for c, l in zip(colors_bins, labels_bins)
]


# ===============================
# Case-study stations icon
# ===============================

class PinLegend(Line2D):

    def __init__(self):
        super().__init__(
            [],
            [],
            label="Case-study stations"
        )


class HandlerPin(HandlerBase):

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans
    ):

        imagebox = OffsetImage(
            pin_image,
            zoom=0.008,
            interpolation="lanczos"
        )

        pin_artist = AnnotationBbox(
            imagebox,
            (
                xdescent + width / 2,
                ydescent + height / 2 - 2
            ),
                        xycoords=trans,
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0
        )

        return [pin_artist]


pin_handle = PinLegend()


# ===============================
# Combine all legend elements
# ===============================

legend_handles = (
    legend_markers +
    [section_handle] +
    legend_bins +
    [pin_handle]
)


# ===============================
# Single legend
# ===============================

leg = ax.legend(
    handles=legend_handles,

    loc="lower left",
    bbox_to_anchor=(-0.002, -0.01),
    bbox_transform=ax.transAxes,

    ncol=1,

    frameon=True,
    fancybox=False,
    framealpha=0.95,
    facecolor="white",
    edgecolor="lightgray",

    fontsize=6,
    borderpad=0.35,
    labelspacing=0.20,
    handlelength=1.0,
    handleheight=1.0,
    handletextpad=0.45,
    columnspacing=0.8,

    handler_map={
        PinLegend: HandlerPin(),
        SectionHeader: HandlerSectionHeader()
    }
)


leg.get_frame().set_linewidth(0.5)

# ===============================
# Layout
# ===============================

plt.tight_layout(
    rect=[0, 0.1, 1, 1]
)


# ===============================
# Save figure
# ===============================

fig.canvas.draw()

fig.savefig(
    path_fig / "fig01.pdf",
    bbox_inches="tight",
    dpi=900
)

plt.show()

#%%