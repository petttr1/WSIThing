# build stage
FROM node:16 as build-stage
RUN apt-get clean && apt-get update -qq && apt-get install -y bash git openssh-server
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# production stage
FROM nginx:stable-alpine as production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]