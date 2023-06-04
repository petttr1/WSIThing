<template>
  <div class="WSICont">
    <div v-show="showWSI">
      <WSI :imgData="WSIParams" :overlays="WSIOverlays"></WSI>
    </div>
    <div v-show="!showWSI">
      <p style="margin-bottom: auto; margin-top: auto">Please Upload a WSI</p>
    </div>
  </div>
</template>

<script>
import WSI from "@/components/WSI.vue";

export default {
  name: "Home",
  components: {
    WSI,
  },
  data() {
    return {
      WSIParams: {},
      WSIOverlays: {},
      showWSI: false,
    };
  },
  mounted() {
    this.$root.$on("NewWSI", this.loadNewWSI);
    this.$root.$on("NewOverlay", this.loadNewOverlay);
  },
  methods: {
    loadNewWSI(data) {
      this.WSIParams = data;
      this.showWSI = true;
    },
    loadNewOverlay(data) {
      this.WSIOverlays = data;
    },
  },
};
</script>
