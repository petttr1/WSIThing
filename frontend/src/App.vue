<template>
  <div id="app">
    <b-navbar toggleable="lg" type="dark" variant="info">
      <b-navbar-brand>WSI Tools</b-navbar-brand>
      <b-button v-b-toggle.sidebar-1>New WSI</b-button>
      <b-navbar-nav class="ml-auto">
        <b-button v-b-toggle.sidebar-2>WSI Options</b-button>
      </b-navbar-nav>
    </b-navbar>
    <router-view />
    {{ message }}
    <b-sidebar id="sidebar-1" title="File Upload" backdrop shadow width="500px">
      <b-form style="width: 70%; margin: auto">
        <b-form-group label="WSI File" label-cols-sm="2" label-size="sm">
          <b-form-file
            v-model="file"
            id="file-large"
            ref="file-input"
            size="sm"
          ></b-form-file>
        </b-form-group>

        <b-form-group label="WSI Annotation" label-cols-sm="2" label-size="sm">
          <b-form-file
            v-model="annot"
            id="file-annot"
            ref="file-input"
            size="sm"
          ></b-form-file>
        </b-form-group>

        <b-form-group
          label="WSI Analysis Export"
          label-cols-sm="2"
          label-size="sm"
        >
          <b-form-file
            v-model="analysisExport"
            id="file-export"
            ref="file-input"
            size="sm"
          ></b-form-file>
        </b-form-group>

        <b-button v-b-toggle.sidebar-1 v-on:click="upload" variant="success"
          >Submit</b-button
        >
      </b-form>
      <h2 class="mt-2">Demo Slides</h2>
      <b-button
        v-for="slide in demo_slides"
        :key="slide"
        @click="loadDemoSlide(slide)"
        >{{ slide }}</b-button
      >
    </b-sidebar>

    <b-sidebar id="sidebar-2" title="WSI Options" shadow right width="500px">
      <!-- <OverlayPicker analysisName="analysis1"></OverlayPicker> -->
      <div id="Analyses">
        <OverlayPicker analysisName="default" :storageKey="storageKey">
        </OverlayPicker>
        <OverlayPicker
          v-for="option in analysesOpts"
          :key="option.value"
          :analysisName="option.value"
          :storageKey="storageKey"
        ></OverlayPicker>
      </div>
    </b-sidebar>
  </div>
</template>

<script>
import OverlayPicker from "@/components/OverlayPicker.vue";
export default {
  components: {
    OverlayPicker,
  },
  data() {
    return {
      message: "",
      file: null,
      annot: null,
      analysisExport: null,
      analysesOpts: [],
      storageKey: "",
      demo_slides: [],
    };
  },
  methods: {
    getDemoSlides() {
      this.$http
        .get(`http://127.0.0.1:5000/get_demo_slides`, {
          headers: {
            "Content-Type": "application/json",
          },
          params: {},
        })
        .then((res) => {
          // eslint-disable-next-line
          console.log(res);
          this.demo_slides = res.data.slides;
        });
    },
    loadDemoSlide(name) {
      this.$http
        .get(`http://127.0.0.1:5000/demo_slide/${name}`, {
          headers: {
            "Content-Type": "application/json",
          },
          params: {},
        })
        .then((res) => {
          // eslint-disable-next-line
          console.log(res);
        });
    },
    uploadWSI() {
      const data = new FormData();
      data.append("files", this.file, this.file.name);
      data.append("files", this.annot, this.annot.name);
      data.append("files", this.analysisExport, this.analysisExport.name);
      // eslint-disable-next-line
      console.log(JSON.stringify(data), typeof data);
      this.$http
        .post("http://127.0.0.1:5000/upload_wsi", data, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((res) => {
          // eslint-disable-next-line
          console.log(res);
        });
    },
    upload(e) {
      // update the message state
      this.message = "Uploading...";
      this.uploadWSI();

      // stop the event from propagating so route won't change
      e.preventDefault();
      e.stopPropagation();
    },
    connectToSocket() {
      this.$socket.emit("connect");
    },
  },
  mounted() {
    this.connectToSocket();
    this.getDemoSlides();
  },
  sockets: {
    statusMessage(data) {
      this.message = data;
    },
    annotationProcessed(data) {
      const Annot = data.annot;
      // eslint-disable-next-line
      console.log(JSON.stringify(Annot));
      this.$root.$emit("NewOverlay", Annot);
    },
    wsiProcessed(data) {
      const WSIForm = data.wsi;
      // eslint-disable-next-line
      console.log(JSON.stringify(WSIForm));
      this.storageKey = WSIForm.Image.storageKey;
      this.$root.$emit("NewWSI", WSIForm);
    },
    analyzerCreated(data) {
      const analyses = data;
      for (var a in analyses) {
        this.analysesOpts.push({
          text: analyses[a],
          value: analyses[a],
        });
      }
    },
    newOverlayLoaded(data) {
      // eslint-disable-next-line
      console.log(JSON.stringify(data.overlay));
      this.$root.$emit("NewOverlay", data.overlay);
    },
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  padding: 30px;
}

#nav a {
  font-weight: bold;
  color: #2c3e50;
}

#nav a.router-link-exact-active {
  color: #42b983;
}
</style>
