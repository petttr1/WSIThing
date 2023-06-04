<template>
  <div class="controls-dropdown">
    <div class="controls-dropdown-header">
      <button @click="$emit('close')">X</button>
    </div>
    <label>Analysis</label>
    <b-form-select
      :options="analysesOpts"
      :value="selected"
      class="mb-4"
      @input="onSelected"
    ></b-form-select>
    <div v-if="selected" class="controls-dropdown-content">
      <label>Mask Opacity ({{ maskOpacity }})</label>
      <b-form-input
        :value="maskOpacity"
        class="mb-4"
        max="1"
        min="0"
        step="0.1"
        type="range"
        @input="changeOpacity"
      ></b-form-input>
      <OverlayPicker
        :analysisName="selected"
        :storageKey="storageKey"
        class="mb-4"
        @update:thresh="onThreshUpdate"
      />
      <label>Class Display</label>
      <ColorPicker
        v-for="color in colors"
        :key="color.name"
        :color="color"
        :visible="isClassVisible(color.name)"
        @update:color="updateColor"
        @toggle:class="toggleClass"
      />
      <button
        v-if="selected"
        class="controls-dropdown-content-reset-colors"
        @click="resetColors"
      >
        Reset colors
      </button>
    </div>
  </div>
</template>

<script>
import OverlayPicker from "./OverlayPicker.vue";
import { mapGetters } from "vuex";
import ColorPicker from "@/components/ColorPicker.vue";

export default {
  name: "ControlsDropdown",
  components: { ColorPicker, OverlayPicker },
  props: ["analysesOpts"],
  data() {
    return {};
  },
  mounted() {},
  computed: {
    ...mapGetters(["storageKey", "selected", "thresh", "maskOpacity"]),
    colors() {
      const c = this.$store.getters.colors(this.selected);
      return Object.keys(c).map((key) => ({ name: key, color: c[key] }));
    },
  },
  methods: {
    isClassVisible(className) {
      const visible = this.$store.getters.classesVisible(this.selected)[
        className
      ];
      console.log("visible", visible, className);
      return visible;
    },
    onSelected(value) {
      this.$root.$emit("removeOverlay", this.selected);
      this.$store.dispatch("updateOptions", { selected: value });
      this.getAnalysis();
    },
    getAnalysis() {
      console.log("get an", this.storageKey, this.selected, this.thresh);
      this.$http
        .get(
          `http://127.0.0.1:5000/get_analysis/${this.storageKey}/${this.selected}`,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
            params: {
              threshold: this.thresh,
            },
          }
        )
        .then((res) => {
          // eslint-disable-next-line
            console.log(res);
        });
    },
    onThreshUpdate(value) {
      this.$store.dispatch("updateOptions", { thresh: value });
      this.$root.$emit("removeOverlay", this.selected);
      this.getAnalysis();
    },
    changeOpacity(value) {
      this.$store.dispatch("updateOptions", { maskOpacity: value });
      this.$root.$emit("changeOpacity", this.selected);
    },
    updateColor(payload) {
      const c = this.$store.getters.colors(this.selected);
      const merged = { ...c, ...payload };
      this.$store.dispatch("storeColors", {
        id: this.selected,
        colors: merged,
      });
      this.$root.$emit("changeColor", this.selected, Object.keys(payload)[0]);
    },
    resetColors() {
      this.$root.$emit("resetColors", this.selected);
    },
    toggleClass(payload) {
      console.log("toggle", payload);
      const v = this.$store.getters.classesVisible(this.selected);
      const merged = { ...v, ...payload };
      this.$store.dispatch("storeVisibility", {
        id: this.selected,
        visible: merged,
      });
      this.$root.$emit(
        "changeVisibility",
        this.selected,
        Object.keys(payload)[0]
      );
    },
  },
};
</script>

<style scoped>
label {
  color: black;
}

.controls-dropdown {
  width: 100%;
  position: fixed;
  top: 64px;
  right: 16px;
  max-width: 250px;
  background: #a1a1a1;
  border-radius: 8px;
  padding: 8px;
}

.controls-dropdown-header {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: flex-end;
  margin-bottom: 8px;
}

.controls-dropdown-header > button {
  color: white;
  background: gray;
  padding: 4px;
  border-radius: 4px;
  border: none;
  width: 32px;
  height: 32px;
}

.controls-dropdown-header > button:hover {
  background: black;
}

.controls-dropdown-content {
  width: 100%;
  height: 100%;
  overflow-y: auto;
  max-height: 500px;
}
</style>
