<template>
  <div class="color-picker">
    <div class="color-picker-top">
      <div class="color-picker-top-left">
        <input
          :id="`check-${color.name}`"
          :checked="visible"
          type="checkbox"
          @input="toggleClass"
        />
        <div class="color-picker-name">{{ color.name }}</div>
      </div>
      <button
        v-if="!picking"
        :style="{ 'background-color': color.color }"
        class="color-picker-color"
        @click="openPicker"
      ></button>
      <input
        v-else
        :ref="`color-picker-${this.color.name}`"
        v-model="colorCode"
        @blur="accept"
      />
    </div>
    <div class="color-picker-bottom">
      <label>Class Weight ({{ classWeight }})</label>
      <b-form-input
        :value="classWeight"
        max="10"
        min="0"
        step="0.1"
        type="range"
        @input="changeWeight"
      ></b-form-input>
    </div>
  </div>
</template>

<script>
export default {
  name: "ColorPicker",
  components: {},
  props: ["color", "visible", "classWeight"],
  data() {
    return {
      picking: false,
      colorCode: "",
    };
  },
  mounted() {
    this.colorCode = this.color.color;
    console.log("visible", this.color.name, this.visible);
  },
  computed: {},
  methods: {
    accept() {
      const payload = {};
      payload[this.color.name] = this.colorCode;
      this.$emit("update:color", payload);
      this.picking = false;
      this.colorCode = "";
    },
    toggleClass() {
      const payload = {};
      payload[this.color.name] = !this.visible;
      this.$emit("toggle:class", payload);
    },
    openPicker() {
      this.picking = true;
    },
    changeWeight() {},
  },
};
</script>

<style scoped>
label {
  color: black;
  text-align: left;
  width: 100%;
}

.color-picker {
  width: 100%;
  margin: 0 0 8px;
  padding: 8px 16px;
}

.color-picker-top {
  width: 100%;

  display: flex;
  justify-content: space-between;
  align-items: center;
}

.color-picker-top-left {
  width: 100%;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  gap: 8px;
}

.color-picker-color {
  max-width: 32px;
  height: 16px;
  border: none;
  border-bottom: 1px solid darkgray;
  border-radius: 4px;
  width: 100%;
}

.color-picker-name {
  color: black;
  text-align: left;
}
</style>
