<template>
  <div id="pick">
    <label>Confidence threshold ({{ this.thresh }})</label>
    <b-form-input
      v-model="thresh"
      max="1"
      min="0"
      step="0.1"
      type="range"
      @input="switchState"
    ></b-form-input>
  </div>
</template>

<script>
export default {
  name: "OverlayPicker",
  components: {},
  props: ["analysisName", "storageKey"],
  data() {
    return {
      thresh: 0.4,
    };
  },
  mounted() {},
  methods: {
    switchState() {
      this.$emit("update:thresh", this.thresh);
    },
  },
};
</script>

<style scoped>
label {
  color: black;
}
</style>
