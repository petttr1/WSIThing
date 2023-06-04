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
      palette: [
        "#FF0000",
        "#0000FF",
        "#00FF00",
        "#00FFFF",
        "#FF00FF",
        "#b22222",
        "#FFFF00",
      ],
    };
  },
  mounted() {
    this.loadSeaDragon();
    this.$root.$on("changeOpacity", this.changeOverlayOpacity);
    this.$root.$on("removeOverlay", this.removeOverlay);
    this.$root.$on("changeColor", this.changeClassColor);
    this.$root.$on("changeVisibility", this.changeClassVisible);
    this.$root.$on("resetColors", this.resetColors);
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
        let colors = this.$store.getters.colors(newVal.id);
        let visible = this.$store.getters.classesVisible(newVal.id);
        if (!colors || !colors.length) {
          colors = this.getColors(newVal.classes);
          this.$store.dispatch("storeColors", {
            id: newVal.id,
            colors,
          });
        }
        if (!visible || !visible.length) {
          visible = this.getVis(newVal.classes);
          this.$store.dispatch("storeVisibility", {
            id: newVal.id,
            visible,
          });
        }
        // eslint-disable-next-line
        console.log("NewVal", newVal, colors);
        this.addOverlay(newVal.objects, newVal.id, colors);
      },
    },
  },
  methods: {
    getVis(classes) {
      let c = {};
      for (let i = 0; i < classes.length; i++) {
        c[classes[i]] = true;
      }
      return c;
    },
    getColors(classes) {
      let c = {};
      for (let i = 0; i < classes.length; i++) {
        c[classes[i]] = this.palette[i];
      }
      return c;
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
      if (typeof this.viewer.world.getItemAt(0) !== "undefined") {
        const opacity = this.$store.getters.maskOpacity;
        polys.forEach((poly) => {
          const className = poly.class.toLowerCase().replaceAll(" ", "-");
          const polyString = this.makePolyString(poly.points.x, poly.points.y);
          // eslint-disable-next-line
          console.log('class', poly.class, colors[poly.class]);
          d3.select(this.overlay.node())
            .append("polygon")
            .attr("points", polyString)
            .style("fill", colors[poly.class])
            .style("opacity", opacity)
            .attr("id", id)
            .attr("class", className);
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
    changeOverlayOpacity(overlayId) {
      const opacity = this.$store.getters.maskOpacity;
      d3.select(this.overlay.node())
        .selectAll("#" + overlayId)
        .style("opacity", opacity);
    },
    changeClassColor(overlayId, className) {
      const color = this.$store.getters.colors(overlayId)[className];
      d3.select(this.overlay.node())
        .selectAll(`.${className.toLowerCase().replaceAll(" ", "-")}`)
        .style("fill", color);
    },
    changeClassVisible(overlayId, className) {
      const visibility =
        this.$store.getters.classesVisible(overlayId)[className];
      console.log("toggle WIS", visibility, className);
      d3.select(this.overlay.node())
        .selectAll(`.${className.toLowerCase().replaceAll(" ", "-")}`)
        .attr("hidden", visibility ? null : true);
    },
    resetColors(id) {
      const classes = Object.keys(this.$store.getters.colors(id));
      const colors = this.getColors(classes);
      this.$store.dispatch("storeColors", { id, colors });
      classes.forEach((analysisClass) => {
        this.changeClassColor(id, analysisClass);
      });
    },
  },
  created() {},
};
</script>
