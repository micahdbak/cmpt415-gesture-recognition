#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#define SHM_NAME "/theta_stream"

// each frame is 1920 (w) x 960 (h) x 3 (RGB channels) bytes in size
#define SHBUFFER_SIZE (1920 * 960 * 3)

int sigint_received = 0;

#define INTERRUPT_MSG\
    "\nInterrupting...\n"

void handle_sigint(int sig) {
    write(STDOUT_FILENO, INTERRUPT_MSG, sizeof(INTERRUPT_MSG));
    sigint_received = 1;
}

int main(int argc, char *argv[]) {
    GstElement *pipeline, *appsink;
    GstBus *bus;
    GstMessage *msg;
    GstSample *sample;
    GstBuffer *buffer;
    GstMapInfo map;
    GstCaps *caps;
    int fd = 0;
    void *shbuffer = NULL;

    // handle sigint
    signal(SIGINT, handle_sigint);

    // prepare shared buffer
    if ((fd = shm_open(SHM_NAME, O_CREAT|O_RDWR, 0666)) == -1) {
        perror("shm_open");
	return EXIT_FAILURE;
    }

    if (ftruncate(fd, SHBUFFER_SIZE) == -1) {
        perror("ftruncate");
	return EXIT_FAILURE;
    }

    shbuffer = mmap(NULL, SHBUFFER_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (shbuffer == MAP_FAILED) {
        perror("mmap");
        return EXIT_FAILURE;
    }

    gst_init(&argc, &argv);

    pipeline = gst_parse_launch(
        "thetauvcsrc mode=2K ! queue max-size-buffers=1 ! h264parse ! avdec_h264 "
	"! videoconvert ! video/x-raw,format=RGB ! appsink name=sink sync=false",
        NULL
    );

    if (!pipeline) {
        g_printerr("Failed to create pipeline\n");
        return EXIT_FAILURE;
    }

    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    gst_app_sink_set_emit_signals((GstAppSink*)appsink, TRUE);
    gst_app_sink_set_drop((GstAppSink*)appsink, TRUE);
    gst_app_sink_set_max_buffers((GstAppSink*)appsink, 1);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    printf("Streaming to %s...\n", SHM_NAME);

    // perform until CTRL+C
    while (!sigint_received) {
        sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
        if (!sample) {
            g_printerr("Failed to get sample\n");
            break;
        }

        buffer = gst_sample_get_buffer(sample);
        gst_buffer_map(buffer, &map, GST_MAP_READ);
	memcpy(shbuffer, map.data, SHBUFFER_SIZE);
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
    }

    munmap(shbuffer, SHBUFFER_SIZE);
    close(fd);
    shm_unlink(SHM_NAME);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}

