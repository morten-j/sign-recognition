FROM node:lts

WORKDIR /nextjs/

COPY package*.json .
RUN npm install

COPY . .

CMD npm run dev