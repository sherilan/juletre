

#include <WiFi.h>
#include <Adafruit_NeoPixel.h>

#define PIN_NEO_PIXEL 2  // The ESP32 pin GPIO16 connected to NeoPixel
#define NUM_PIXELS 400     // The number of LEDs (pixels) on NeoPixel LED strip
#define BUFFER_SIZE 30 // Should be >= 2
#define FPS 60 // 30 //60


#define MODE_PREAMBLE 99
#define MODE_DENSE 1
#define MODE_SPARSE 2 
#define MODE_FILL 3

struct Color
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct SparseColor
{
  uint16_t pixel;
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct DenseHeader
{
  uint16_t offset;
  uint16_t length;
  uint16_t stride;
};

struct SparseHeader
{
  uint16_t length;
};

struct FillHeader
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
};



struct LEDBuffer {
  SemaphoreHandle_t lock = NULL;
  Color colors[NUM_PIXELS];
  bool updated[NUM_PIXELS];
  bool acquireLock(TickType_t waittime = portMAX_DELAY) {
    return xSemaphoreTake(lock, waittime) == pdTRUE;
  }
  void releaseLock() {
    xSemaphoreGive(lock);
  }
  void clearUpdated(bool value = false) {
    memset(&updated, value, sizeof(updated));
  }
};

LEDBuffer buffers[BUFFER_SIZE];
size_t global_buffer_index;
size_t local_buffer_index;
int FRAME = 0;
unsigned long last_request_time_ms = 0;
char IDLE_STATE[NUM_PIXELS];

// Initialize LEDs
Adafruit_NeoPixel neo_pixels(NUM_PIXELS, PIN_NEO_PIXEL, NEO_RGB + NEO_KHZ800);


// const char* ssid = "";     // Replace with your Wi-Fi SSID
// const char* password = ""; // Replace with your Wi-Fi password

const char* ssid = "JuletreESP";
const char* password = "godjul";

WiFiServer server(
  80, // Server will run on port 80
  1   // Maximum one client at the time 
); 

struct {
  SemaphoreHandle_t lock = NULL;
  WiFiClient* client;
  unsigned long id = 0;
  bool acquireLock(TickType_t waittime = portMAX_DELAY) {
    return xSemaphoreTake(lock, waittime) == pdTRUE;
  }
  void releaseLock() {
    xSemaphoreGive(lock);
  }
} g_conn;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Set up pixels 
  neo_pixels.begin();  // initialize NeoPixel strip object (REQUIRED)

  WiFi.softAP(ssid);
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  // // Connect to Wi-Fi
  // WiFi.begin(ssid, password);
  // Serial.print("Connecting to WifI ");
  // while (WiFi.status() != WL_CONNECTED) {
  //   delay(500);
  //   Serial.print(".");
  // }
  // Serial.println(" OK!");

  last_request_time_ms = 0;
  for (int i = 0; i < NUM_PIXELS; i++)
  {
    IDLE_STATE[i] = random(0, 255);
  }

  g_conn.lock = xSemaphoreCreateMutex();

  
  // Initialize the mutexe
  for (int i = 0; i < BUFFER_SIZE; i++){
    Serial.printf("Creating semaphore %i\n", i);
    buffers[i].lock = xSemaphoreCreateMutex();
  }
  // Grab the first semaphore in the main thread
  global_buffer_index = 0;
  

  // Create illumination task (for setting led values)
  static int task_number0 = 0;
  // xTaskCreate(
  //   illuminate
  //   ,  "Task 0" // A name just for humans
  //   ,  2048          // The stack size
  //   ,  (void*)&task_number0 // Pass reference to a variable describing the task number
  //   //,  5  // High priority
  //   ,  1  // priority
  //   ,  NULL // Task handle is not used here - simply pass NULL
  // );
  

  // Start the server
  server.begin();
  server.setNoDelay(true);
  Serial.println("TCP Server started");
  Serial.print("Server IP: ");
  Serial.println(WiFi.localIP());
  Serial.printf("Server nodelay: %i\n", server.getNoDelay());
  Serial.print("Arduhal loglevel: "); Serial.println(ARDUHAL_LOG_LEVEL);
  Serial.print("Setup on core: "); Serial.println(xPortGetCoreID());
  Serial.print("Refresh rate: "); Serial.print(FPS); Serial.println(" Hz.");

  
  xTaskCreatePinnedToCore(
    communicate, /* Function to implement the task */
    "communicate", /* Name of the task */
    2048,  /* Stack size in words */
    NULL,  /* Task input parameter */
    1,  /* Priority of the task (5=HIGH) */
    NULL, // &Task1,  /* Task handle. */
    1 //
  );
  delay(10);
  xTaskCreatePinnedToCore(
    illuminate, /* Function to implement the task */
    "Illuminate", /* Name of the task */
    2048,  /* Stack size in words */
    NULL,  /* Task input parameter */
    5,  /* Priority of the task (5=HIGH) */
    NULL, // &Task1,  /* Task handle. */
    1  /* Core where the task should run (NOTE: setup/loop runs at 1 by default) */
  );
  Serial.println("Started tasks!");


}

bool awaitClientData(WiFiClient &client, uint patience_us = 1000000, uint sleep_us = 10)
{
  for (int i = 0; i < patience_us / sleep_us; i++)
  {
    if (client.available())
    {
      return true;
    }
    if (!client.connected())
    {
      Serial.println("Client disconnected");
      return false;
    }
    delayMicroseconds(10);
  }
  Serial.println("Timeout when waiting for client");
  return false;
}

bool readBytes(WiFiClient &client, uint8_t *buf, size_t len, uint patience_us = 1000000, uint sleep_us = 10)
{
  for (int i = 0; i < patience_us / sleep_us; i++)
  {
    size_t available = client.available();
    if (available >= len)
    {
      size_t nread = client.read(buf, len);
      if (nread != len)
      {
        Serial.print("Read wrong number of bytes. Expected ");
        Serial.print(len);
        Serial.print(" but ended up reading ");
        Serial.print(nread);
        Serial.println(".");
      }
      return true;

    }
    if (!client.connected())
    {
      Serial.println("Client disconnected");
      return false;
    }
    delayMicroseconds(10);
  }
  Serial.println("Timeout when waiting for client");
  return false;
}

template <typename T>
bool read(WiFiClient &client, T &data, uint patience_us = 1000000, uint sleep_us = 10)
{
  return readBytes(client, (uint8_t *)&data, sizeof(T), patience_us, sleep_us);
}


bool parse_led_message(WiFiClient &client, LEDBuffer &buffer)
{
  buffer.clearUpdated();
  uint8_t mode = MODE_PREAMBLE;
  while (mode == MODE_PREAMBLE)
  {
    if (!read(client, mode))
    {
      Serial.println("Failed to read preamble!");
      return false;
    }
  }

  if (mode == MODE_DENSE)
  {
    // Serial.println("Executing dense mode!");
    DenseHeader header;
    if (!read(client, header))
    {
      Serial.println("Failed to read Dense header");
      return false;
    }
    Color color;
    for (int i = 0; i < header.length; i ++)
    {
      if (!read(client, color))
      {
        Serial.print("Failed to read dense color: ");
        Serial.println(i);
        return false;
      }
      int pixel = header.offset + i * header.stride;
      if (pixel >= 0 && pixel < NUM_PIXELS)
      {
        buffer.colors[pixel] = color;
        buffer.updated[pixel] = true;
        // NeoPixel.setPixelColor(pixel, NeoPixel.Color(color.r, color.g, color.b));
      }
      // Serial.printf("%i) Setting pixel %i to {%i %i %i}\n", i, pos, color.r, color.g, color.b);
    }

    // Serial.printf("Successfully set dense colors for frame %i!\n", FRAME);
    // client.println("Thanks for the bytes!");
  }
  else
  {
    Serial.print("Bad mode byte received: ");
    Serial.println(mode);
    return false;
  }
  // long t0 = millis();
  // NeoPixel.show();
  // long t1 = millis();
  // Serial.printf("Lit LEDs in %i ms.\n", t1 - t0);
  return true; // TODO set leds and return true instead 
}

void loop()
{
  delay(10);
}
void communicate(void *pvParameters)
{
  Serial.print("Communicate running on core: "); Serial.println(xPortGetCoreID());
  if (buffers[global_buffer_index].acquireLock(1000 / portTICK_PERIOD_MS)) {
    Serial.println("Grabbed the first semaphore");
  } else {
    Serial.println("Failed to grab the first semaphore :(");
  }
  while (true)
  {
    communicate_loop();
  }
}
void communicate_loop() {
  // put your main code here, to run repeatedly:
  WiFiClient client = server.available();
  if (client) {
    // New client connected with TCP socket 
    Serial.print("Connected to: "); Serial.println(client.remoteIP());

    g_conn.acquireLock();
    g_conn.client = &client;
    g_conn.id++;
    g_conn.releaseLock();

    // Reset frame counter 
    FRAME = 0;

    // Begin communication loop 
    while (1)
    {
      // Update last request time so we know when to idle again 
      last_request_time_ms = millis();

      // Start timer 
      unsigned long t0 = millis();

      // Acquire semaphore for next buffer 
      size_t next_buffer_index = (global_buffer_index + 1) % BUFFER_SIZE;
      buffers[next_buffer_index].acquireLock();

      // Parse next frame and write to current buffer 
      bool success = parse_led_message(client, buffers[global_buffer_index]);
      FRAME++;

      // Release lock for current buffer index and advance index 
      buffers[global_buffer_index].releaseLock();
      global_buffer_index = next_buffer_index;

      // // Let client know that we received the command
      // client.print(1);
      
      // Stop if communication broke down 
      if (!success) { break; }

      // Stats 
      int diff = int(global_buffer_index) - int(local_buffer_index);
      diff = diff < 0 ? BUFFER_SIZE + diff : diff;
      unsigned long t1 = millis();
      unsigned long el = t1 - t0;
      Serial.printf("Finished frame in %i ms. Current diff: %i  avail=%i\n", el, diff, client.available());

      
    }

    // Shut down socket connection 
    g_conn.acquireLock();
    Serial.println("Disconnecting");
    client.stop();
    g_conn.client = NULL;
    g_conn.releaseLock();
  }
  else
  {
    delay(1);
    // server.stopAll();
    // server.clearWriteError();
    // Serial.println("Server clean");
    unsigned long now = millis();
    if (
      last_request_time_ms == 0 || // On startup
      now - last_request_time_ms > 5000 ||  // After 5 sec delay 
      last_request_time_ms > now // Wraparound (~50 days I think, but just in case)
    )
    {
      idle();
      delay(10);
    }
  }
}

int set_led_colors(const LEDBuffer &buffer) {
  int num_set = 0;
  for (size_t pixel = 0; pixel < NUM_PIXELS; pixel++) {
    if (!buffer.updated[pixel]) { continue; }
    const Color *c = &(buffer.colors[pixel]);
    neo_pixels.setPixelColor(pixel, c->r, c->g, c->b);
    num_set++;
  }
  return num_set;
}

void illuminate(void *pvParameters) {
  Serial.print("Illuminate running on core: "); Serial.println(xPortGetCoreID());
  local_buffer_index = 0;
  buffers[local_buffer_index].acquireLock();
  unsigned long dt_target_ms = 1000 / FPS;
  while (1)
  {
    // Start timer 
    unsigned long t0_ms = millis();

    // Grab semaphore for next buffer 
    size_t next_buffer_index = (local_buffer_index + 1) % BUFFER_SIZE;
    buffers[next_buffer_index].acquireLock();

    // Update LEDs based on data from current buffer 
    int num_set = set_led_colors(buffers[next_buffer_index]);
    neo_pixels.show();

    // Notify that colors has been shown 
    g_conn.acquireLock();
    if (g_conn.client) { g_conn.client->print(1); }
    g_conn.releaseLock();

    // Release lock on current buffer 
    buffers[local_buffer_index].releaseLock();
    local_buffer_index = next_buffer_index;

    // Record time and sleep to maintain target fps 
    unsigned long t1_ms = millis();
    unsigned long dt_ms = t1_ms - t0_ms;
    Serial.printf("Updated %i LEDs in %i ms\n", num_set, dt_ms);
    if (dt_ms >= 0 and dt_ms < dt_target_ms) { delay(dt_target_ms - dt_ms); }
  }
}

void idle()
{
  double circle_norm = 2 * M_PI / 256;
  // Serial.print("IDLE: ");
  for (int i = 0; i < NUM_PIXELS; i++)
  {
    double p = double(IDLE_STATE[i]) * circle_norm;
    double a = sin(p);
    // int h = 5000; //50000; //65535;
    // int s = 200;
    // int v = 50 + int(a * 50);
    // neo_pixels.setPixelColor(i, neo_pixels.ColorHSV(h, s, v));
    uint8_t r = 255;
    uint8_t g = 50 + int(a * 50);
    uint8_t b = 0;
    neo_pixels.setPixelColor(i, r, g, b);
    IDLE_STATE[i] = (IDLE_STATE[i] + 1) % 256;
    // if (i < 5)
    // {
    //   Serial.print(v);
    //   Serial.print(", ");
    // }
  }
  // Serial.println();
  neo_pixels.show();
  // neo_pixels.ColorHSV(uint16_t hue)
}
