<template>
  <div id="pick" class="mb-2">
    <b-card header-tag="header">
      <template #header>
        <b-form-checkbox
          @input="switchState"
          v-model="checked"
          name="check-button"
          switch
          v-if="analysisName != 'default'"
        >
          {{ analysisName }}
        </b-form-checkbox>
        <p v-else>
          {{ analysisName }}
        </p>
      </template>
      <div v-if="checked || analysisName == 'default'">
        <label>Opacity ({{ this.opacity }})</label>
        <b-form-input
          v-model="opacity"
          type="range"
          min="0"
          max="1"
          step="0.1"
          @input="changeOpacity"
        ></b-form-input>
        <div v-if="analysisName != 'default'">
          <label>Thresh level ({{ this.thresh }})</label>
          <b-form-input
            v-model="thresh"
            type="range"
            min="0"
            max="1"
            step="0.1"
            @input="switchState"
          ></b-form-input>
        </div>
      </div>
    </b-card>
  </div>
</template>

<script>
export default {
  name: "OverlayPicker",
  components: {},
  props: ["analysisName", "storageKey"],
  data() {
    return {
      checked: false,
      opacity: 0.5,
      thresh: 0.4,
    };
  },
  mounted() {},
  methods: {
    switchState() {
      if (this.checked) {
        this.getAnalysis();
      } else {
        this.$root.$emit("removeOverlay", this.analysisName);
      }
    },
    getAnalysis() {
      this.$http
        .get(
          `http://127.0.0.1:5000/get_analysis/${this.storageKey}/${this.analysisName}`,
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
    changeOpacity() {
      // eslint-disable-next-line
      console.log(this.analysisName, this.opacity);
      this.$root.$emit("changeOpacity", this.analysisName, this.opacity);
    },
  },
};
</script>

<style scoped>
#pick {
  border-style: solid;
  border-width: 2px;
  border-radius: 5px;
  border-color: darkslategrey;
}
</style>