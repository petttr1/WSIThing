<template>
  <div class="WSI">
    <div
      id="openseadragon"
      style="width: 95%; height: 700px; border: 1px solid red; margin: auto"
    ></div>
  </div>
</template>

<script>
import OpenSeadragon from "openseadragon";
import "svg-overlay";
import * as d3 from "d3";
export default {
  name: "WSI",
  components: {},
  props: {
    imgData: Object,
    overlays: Object,
  },
  data() {
    return {
      tileSources: {},
      viewer: null,
      overlay: null,
      activeOverlays: {},
      palette: ["red", "blue", "green", "cyan", "fuchsia", "aqua", "yellow"],
    };
  },
  mounted() {
    this.loadSeaDragon();
    this.$root.$on("changeOpacity", this.changeOverlayOpacity);
    this.$root.$on("removeOverlay", this.removeOverlay);
  },
  watch: {
    imgData: {
      deep: true,
      handler(newVal) {
        this.tileSources = newVal;
        this.viewer.open(this.tileSources);
      },
    },
    overlays: {
      deep: true,
      handler(newVal) {
        const colors = this.getColors(newVal.classes);
        // eslint-disable-next-line
        console.log("NewVal", newVal, colors);
        this.addOverlay(newVal.objects, newVal.id, colors);
      },
    },
  },
  methods: {
    getColors(classes) {
      var c = {};
      for (let i = 0; i < classes.length; i++) {
        c[classes[i]] = this.palette[i];
      }
      return c;
      // return this.palette.slice(0, number - 1);
    },
    loadSeaDragon() {
      this.viewer = OpenSeadragon({
        showNavigator: true,
        id: "openseadragon",
        prefixUrl: "//openseadragon.github.io/openseadragon/images/",
        tileSources: this.tileSources,
      });
      this.overlay = this.viewer.svgOverlay();
    },
    makePolyString(xs, ys) {
      var finalString = "";
      for (var i = 0; i < xs.length; i++) {
        const trans_point = this.viewer.world
          .getItemAt(0)
          .imageToViewportCoordinates(xs[i], ys[i]);
        finalString = finalString + `${trans_point.x},${trans_point.y} `;
      }
      return finalString;
    },
    addOverlay(polys, id, colors) {
      // d3.select(this.overlay.node()).selectAll("*").remove();
      if (typeof this.viewer.world.getItemAt(0) !== "undefined") {
        polys.forEach((poly) => {
          const polyString = this.makePolyString(poly.points.x, poly.points.y);
          // eslint-disable-next-line
          console.log(poly.class);
          d3.select(this.overlay.node())
            .append("polygon")
            .attr("points", polyString)
            .style("fill", colors[poly.class])
            .style("opacity", 0.5)
            .attr("id", id)
            .attr("class", poly.class);
        });
      } else {
        setTimeout(this.makePolygons, 1000, polys, id);
      }
    },
    removeOverlay(id) {
      d3.select(this.overlay.node())
        .selectAll("#" + id)
        .remove();
    },
    changeOverlayOpacity(overlayId, opacity) {
      // eslint-disable-next-line
      console.log(overlayId, opacity);
      d3.select(this.overlay.node())
        .selectAll("#" + overlayId)
        .style("opacity", opacity);
    },
  },
  created() {},
};
</script>
