import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";
import { BootstrapVue, IconsPlugin } from "bootstrap-vue";
import axios from "axios";
import VueSocketIO from 'vue-socket.io'

import "bootstrap/dist/css/bootstrap.css";
import "bootstrap-vue/dist/bootstrap-vue.css";

Vue.config.productionTip = false;
Vue.prototype.$http = axios;

Vue.use(BootstrapVue);
Vue.use(IconsPlugin);
Vue.use(new VueSocketIO({
  debug: true,
  connection: 'http://127.0.0.1:5000',
  vuex: {
    store,
    actionPrefix: 'SOCKET_',
    mutationPrefix: 'SOCKET_'
  },
  // options: { path: "/my-app/" } //Optional options
}))

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount("#app");
