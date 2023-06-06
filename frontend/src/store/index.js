import Vue from "vue";
import Vuex from "vuex";

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    viewOptions: {
      selected: null,
      thresh: 0.4,
      maskOpacity: 0.5,
    },
    analysisOptions: {},
    storageKey: "",
  },
  getters: {
    selected(state) {
      return state.viewOptions.selected;
    },
    thresh(state) {
      return state.viewOptions.thresh;
    },
    storageKey(state) {
      return state.storageKey;
    },
    maskOpacity: (state) => {
      return state.viewOptions.maskOpacity;
    },
    colors: (state) => (id) => {
      return state.analysisOptions[id]?.colors ?? null;
    },
    classesVisible: (state) => (id) => {
      return state.analysisOptions[id]?.visible ?? null;
    },
    classesWeights: (state) => (id) => {
      return state.analysisOptions[id]?.weights ?? null;
    },
    classesOpacities: (state) => (id) => {
      return state.analysisOptions[id]?.opacities ?? null;
    },
  },
  mutations: {
    updateState(state, newVal) {
      state.viewOptions = { ...state.viewOptions, ...newVal };
    },
    updateStorageKey(state, newVal) {
      state.storageKey = newVal;
    },
    storeColors(state, payload) {
      const stored = state.analysisOptions[payload.id] ?? {};
      Vue.set(state.analysisOptions, payload.id, {
        ...stored,
        colors: payload.colors,
      });
    },
    storeVisibility(state, payload) {
      const stored = state.analysisOptions[payload.id] ?? {};
      Vue.set(state.analysisOptions, payload.id, {
        ...stored,
        visible: payload.visible,
      });
    },
    storeWeights(state, payload) {
      const stored = state.analysisOptions[payload.id] ?? {};
      Vue.set(state.analysisOptions, payload.id, {
        ...stored,
        weights: payload.weights,
      });
    },
    storeOpacities(state, payload) {
      const stored = state.analysisOptions[payload.id] ?? {};
      Vue.set(state.analysisOptions, payload.id, {
        ...stored,
        opacities: payload.opacities,
      });
    },
  },
  actions: {
    updateOptions(context, value) {
      context.commit("updateState", value);
    },
    updateStorageKey(context, value) {
      context.commit("updateStorageKey", value);
    },
    storeColors(context, payload) {
      context.commit("storeColors", payload);
    },
    storeVisibility(context, payload) {
      context.commit("storeVisibility", payload);
    },
    storeWeights(context, payload) {
      context.commit("storeWeights", payload);
    },
    storeOpacities(context, payload) {
      context.commit("storeOpacities", payload);
    },
  },
  modules: {},
});
