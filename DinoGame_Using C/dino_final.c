#include <ncurses.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define INITIAL_DELAY 50000 // Initial delay in microseconds
#define DINO_HEIGHT 3
#define DINO_WIDTH 5
#define OBSTACLE_WIDTH 2
#define GROUND_Y 30
#define JUMP_HEIGHT 3
#define JUMP_DELAY 8 // Delay in frames for the jump
#define INITIAL_OBSTACLES 2 // Initial number of obstacles
#define MAX_OBSTACLES 6 // Maximum number of obstacles
#define MIN_OBSTACLE_DISTANCE 40 // Minimum distance between obstacles
#define SPEED_INCREASE_INTERVAL 300 // Increase speed every 300 points
#define SPEED_INCREASE_AMOUNT 1500 // Decrease delay by 1500 microseconds every interval
#define OBSTACLE_INCREASE_INTERVAL 1000 // Increase obstacles every 1000 points
#define MIN_DELAY 20000 // Minimum delay to prevent the game from becoming too fast
#define CROUCH_DURATION 10 // Duration for crouching in frames
#define MAX_CLOUDS 5 // Maximum number of clouds

#define OBSTACLE_MIN_HEIGHT GROUND_Y + 1
#define OBSTACLE_MAX_HEIGHT GROUND_Y - 3 // Higher obstacles will be lower on the screen

#define ITEM_SPEED_CHANGE 1
#define ITEM_REVERSE_CONTROLS 2
#define ITEM_TOGGLE_THEME 3

typedef struct {
    int x, y;
} Position;

typedef struct {
    char name[50];
    int score;
    int is_restart; // To indicate if the score is from a restart or a new game
    time_t time;
    char character_name[50]; // To store the selected character name
} Player;

typedef struct {
    char* name;
    char* idle[DINO_HEIGHT];
    char* crouch[DINO_HEIGHT];
} Character;

typedef struct {
    char name[50];
    char character_name[50];
    int score;
    char time_str[26];
} ScoreEntry;

Character characters[5] = {
    {
        "Human",
        {" \\o/", " | ", "/ \\"},
        {" \\o", " | ", "/ \\"}
    },
    {
        "Cat",
        {" /\\_/\\", "( o.o )", " > ^ <"},
        {" /\\_/\\", "( o.o )", " > ^ <"}
    },
    {
        "Dog",
        {"  __", "o-''", "  >"},
        {"  __", "o-''", "  >"}
    },
    {
        "Bird",
        {"  __ ", "/ o\\", "\\_/ "},
        {"  __", "/ o\\", "\\_/ "}
    },
    {
        "Robot",
        {" [o_o]", "/|_|\\", "/_|_\\"},
        {" [o_o]", "/|_|\\", "/_|_\\"}
    }
};

void init_colors() {
    start_color();
    init_pair(1, COLOR_WHITE, COLOR_BLACK); // Normal
    init_pair(2, COLOR_BLACK, COLOR_WHITE); // Inverted
}

void draw_dino(Position dino, int crouching, Character character) {
    if (crouching) {
        mvprintw(dino.y + 1, dino.x, "%s", character.crouch[0]);
        mvprintw(dino.y + 2, dino.x, "%s", character.crouch[1]);
    } else {
        mvprintw(dino.y, dino.x, "%s", character.idle[0]);
        mvprintw(dino.y + 1, dino.x, "%s", character.idle[1]);
        mvprintw(dino.y + 2, dino.x, "%s", character.idle[2]);
    }
}

void draw_obstacle(Position obstacle) {
    mvprintw(obstacle.y, obstacle.x, "##");
}

void draw_cloud(Position cloud) {
    mvprintw(cloud.y, cloud.x, "~~~");
}

void draw_star(Position star) {
    mvprintw(star.y, star.x, "*");
}

void draw_sun() {
    mvprintw(2, COLS - 10, " \\ /");
    mvprintw(3, COLS - 10, "- O -");
    mvprintw(4, COLS - 10, " / \\");
}

void draw_moon() {
    mvprintw(2, COLS - 10, "   _  ");
    mvprintw(3, COLS - 10, "  / \\ ");
    mvprintw(4, COLS - 10, " |  | ");
    mvprintw(5, COLS - 10, "  \\_\\ ");
}

void draw_item(Position item, int item_type) {
    switch (item_type) {
        case ITEM_SPEED_CHANGE:
            mvprintw(item.y, item.x, "[]");
            mvprintw(item.y + 1, item.x, "[]");
            break;
        case ITEM_REVERSE_CONTROLS:
            mvprintw(item.y, item.x, "><");
            mvprintw(item.y + 1, item.x, "><");
            break;
        case ITEM_TOGGLE_THEME:
            mvprintw(item.y, item.x, "**");
            mvprintw(item.y + 1, item.x, "**");
            break;
    }
}

int check_collision(Position dino, Position obstacle, int crouching) {
    int dino_bottom = dino.y + DINO_HEIGHT;
    int dino_top = dino.y;
    if (crouching) {
        dino_bottom = dino.y + DINO_HEIGHT - 1;
        dino_top = dino.y + 1;
    }
    return (dino_bottom > obstacle.y) &&
           (dino_top <= obstacle.y) &&
           (dino.x < obstacle.x + OBSTACLE_WIDTH) &&
           (dino.x + DINO_WIDTH > obstacle.x);
}

int check_item_collision(Position dino, Position item) {
    return (dino.y <= item.y + 1) &&
           (dino.y + DINO_HEIGHT >= item.y) &&
           (dino.x < item.x + 2) &&
           (dino.x + DINO_WIDTH > item.x);
}

void apply_item_effect(int item_type, int *delay, int *reverse_controls, int *theme) {
    switch (item_type) {
        case ITEM_SPEED_CHANGE:
            *delay += (rand() % 2 ? -10000 : 10000);
            if (*delay < MIN_DELAY) *delay = MIN_DELAY;
            if (*delay > 100000) *delay = 100000;
            break;
        case ITEM_REVERSE_CONTROLS:
            *reverse_controls = !(*reverse_controls);
            break;
        case ITEM_TOGGLE_THEME:
            *theme = !(*theme);
            if (*theme) {
                bkgd(COLOR_PAIR(2));
            } else {
                bkgd(COLOR_PAIR(1));
            }
            break;
    }
}

void reset_game(Position *dino, Position obstacles[], Position clouds[], int *score, int *game_over, int *initial_run, int *jumping, int *jump_counter, int *delay, int *max_obstacles, int *last_speed_increase, int *last_obstacle_increase, int *crouching, int *crouch_counter, Position *item, int *item_active, int *reverse_controls, int *theme) {
    dino->x = 5;
    dino->y = GROUND_Y;
    for (int i = 0; i < MAX_OBSTACLES; i++) {
        obstacles[i].x = COLS + i * (MIN_OBSTACLE_DISTANCE + (rand() % MIN_OBSTACLE_DISTANCE));
        obstacles[i].y = GROUND_Y - (rand() % (GROUND_Y - OBSTACLE_MAX_HEIGHT + 1));
        if (obstacles[i].y < OBSTACLE_MAX_HEIGHT) {
            obstacles[i].y = OBSTACLE_MAX_HEIGHT;
        }
        if (obstacles[i].y > OBSTACLE_MIN_HEIGHT) {
            obstacles[i].y = OBSTACLE_MIN_HEIGHT;
        }
    }
    for (int i = 0; i < MAX_CLOUDS; i++) {
        clouds[i].x = rand() % COLS;
        clouds[i].y = rand() % (GROUND_Y / 2);
    }
    *score = 0;
    *game_over = 0;
    *initial_run = 1;
    *jumping = 0;
    *jump_counter = 0;
    *delay = INITIAL_DELAY; // Reset delay to initial value
    *max_obstacles = INITIAL_OBSTACLES; // Reset max obstacles to initial value
    *last_speed_increase = 0; // Reset last speed increase score
    *last_obstacle_increase = 0; // Reset last obstacle increase score
    *crouching = 0; // Reset crouching state
    *crouch_counter = 0; // Reset crouch counter
    *item_active = 0; // Reset item state
    *reverse_controls = 0; // Reset control state
    *theme = 0; // Reset theme to day
    bkgd(COLOR_PAIR(1)); // Set background to normal color
    item->x = -1; // Hide item
    item->y = -1; // Hide item
}

void save_score(Player player) {
    FILE *file = fopen("scores.txt", "a");
    if (file != NULL) {
        char time_str[26];
        struct tm *tm_info;
        tm_info = localtime(&player.time);
        strftime(time_str, 26, "%Y-%m-%d %H:%M:%S", tm_info);
        fprintf(file, "Player: %s, Character: %s, Score: %d, Time: %s\n", player.name, player.character_name, player.score, time_str);
        fclose(file);
    }
}

int check_name_exists(const char* name) {
    FILE *file = fopen("scores.txt", "r");
    if (file == NULL) {
        return 0; // If file does not exist or cannot be opened, return 0
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        char existing_name[50];
        if (sscanf(line, "Player: %49[^,],", existing_name) == 1) {
            if (strcmp(existing_name, name) == 0) {
                fclose(file);
                return 1; // Name exists
            }
        }
    }

    fclose(file);
    return 0; // Name does not exist
}

void get_player_name(Player *player) {
    echo();
    clear();
    mvprintw(LINES / 2 - 2, COLS / 2 - 10, "Enter your name: ");
    getnstr(player->name, 49);
    noecho();

    if (strlen(player->name) == 0) {
        static int unknown_counter = 1;
        char temp_name[50];
        do {
            snprintf(temp_name, 50, "unknown_%d", unknown_counter++);
        } while (check_name_exists(temp_name));
        strncpy(player->name, temp_name, 50);
    }

    // Display instructions
    clear();
    mvprintw(LINES / 2 - 10, COLS / 2 - 20, "Game Instructions:");
    mvprintw(LINES / 2 - 8, COLS / 2 - 20, "1. Use 'Space' or 'Up arrow' to jump.");
    mvprintw(LINES / 2 - 6, COLS / 2 - 20, "2. Use 'Down arrow' to crouch.");
    mvprintw(LINES / 2 - 4, COLS / 2 - 20, "3. Avoid obstacles and collect items.");
    mvprintw(LINES / 2 - 2, COLS / 2 - 20, "4. Press 'p' to pause/resume the game.");

    // Display item descriptions
    mvprintw(LINES / 2, COLS / 2 - 20, "Item Descriptions:");
    mvprintw(LINES / 2 + 2, COLS / 2 - 20, "[]: Changes speed.");
    mvprintw(LINES / 2 + 4, COLS / 2 - 20, "><: Reverses controls.");
    mvprintw(LINES / 2 + 6, COLS / 2 - 20, "**: Switch day/night.");

    mvprintw(LINES / 2 + 8, COLS / 2 - 20, "Press any key to continue.");
    getch();
}

Character select_character(char *selected_character_name) {
    int ch;
    int selected = 0;

    while (1) {
        clear();
        mvprintw(5, 5, "Select your character (Use left/right arrow keys and press Enter):");
        for (int i = 0; i < 5; i++) {
            if (i == selected) {
                attron(A_REVERSE);
            }
            mvprintw(7, 5 + i * 15, characters[i].name);
            mvprintw(8, 5 + i * 15, characters[i].idle[0]);
            mvprintw(9, 5 + i * 15, characters[i].idle[1]);
            mvprintw(10, 5 + i * 15, characters[i].idle[2]);
            attroff(A_REVERSE);
        }
        ch = getch();
        if (ch == KEY_RIGHT) {
            selected = (selected + 1) % 5;
        } else if (ch == KEY_LEFT) {
            selected = (selected + 4) % 5; // 4 instead of -1 to handle negative wrap-around
        } else if (ch == '\n') {
            strcpy(selected_character_name, characters[selected].name);
            break;
        }
        refresh();
    }
    return characters[selected];
}

int read_high_score() {
    FILE *file = fopen("scores.txt", "r");
    if (file == NULL) {
        return 0; // If file does not exist or cannot be opened, return 0
    }

    int high_score = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        int score;
        if (sscanf(line, "Player: %*[^,], Character: %*[^,], Score: %d,", &score) == 1) {
            if (score > high_score) {
                high_score = score;
            }
        }
    }

    fclose(file);
    return high_score;
}

int read_all_scores(int *all_scores) {
    FILE *file = fopen("scores.txt", "r");
    if (file == NULL) {
        return 0; // If file does not exist or cannot be opened, return 0
    }

    int total_scores = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        int score;
        if (sscanf(line, "Player: %*[^,], Character: %*[^,], Score: %d,", &score) == 1) {
            all_scores[total_scores++] = score;
        }
    }

    fclose(file);
    return total_scores;
}

int get_rank(int score, int *all_scores, int total_scores) {
    int rank = 1;
    for (int i = 0; i < total_scores; i++) {
        if (score < all_scores[i]) {
            rank++;
        }
    }
    return rank;
}

int read_scores(ScoreEntry *scores) {
    FILE *file = fopen("scores.txt", "r");
    if (file == NULL) {
        return 0; // If file does not exist or cannot be opened, return 0
    }

    int total_scores = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "Player: %[^,], Character: %[^,], Score: %d, Time: %[^\n]",
                   scores[total_scores].name, scores[total_scores].character_name, &scores[total_scores].score, scores[total_scores].time_str) == 4) {
            total_scores++;
        }
    }

    fclose(file);
    return total_scores;
}

void print_game_over(Player player, int *scores, int try_count) {
    int all_scores[1000]; // Assuming a maximum of 1000 scores in scores.txt
    int total_scores = read_all_scores(all_scores);

    clear();
    mvprintw(5, 0, "Game Over!");
    mvprintw(7, 0, "Final Scores:");

    for (int i = 0; i < try_count; i++) {
        mvprintw(8 + i, 0, "%d try: %d", i + 1, scores[i]);
    }

    mvprintw(9 + try_count, 0, "Ranks:");

    for (int i = 0; i < try_count; i++) {
        // Add current score to the all_scores array to include it in rank calculation
        int current_total_scores = total_scores + i + 1;
        all_scores[current_total_scores - 1] = scores[i];
        int rank = get_rank(scores[i], all_scores, current_total_scores);
        mvprintw(10 + try_count + i, 0, "%d try: Rank %d out of %d\n\n", i + 1, rank, current_total_scores);
    }

    // Print top 10 scores
    ScoreEntry score_entries[1000];
    int score_count = read_scores(score_entries);

    // Sort scores in descending order
    for (int i = 0; i < score_count - 1; i++) {
        for (int j = 0; j < score_count - i - 1; j++) {
            if (score_entries[j].score < score_entries[j + 1].score) {
                ScoreEntry temp = score_entries[j];
                score_entries[j] = score_entries[j + 1];
                score_entries[j + 1] = temp;
            }
        }
    }

    // Display top 10 scores
    mvprintw(15, 0, "Top 10 Scores:");
    mvprintw(16, 0, "+------+-----------------+------------+-------+---------------------+");
    mvprintw(17, 0, "| Rank | Player          | Character  | Score | Time                |");
    mvprintw(18, 0, "+------+-----------------+------------+-------+---------------------+");
    for (int i = 0; i < score_count && i < 10; i++) {
        mvprintw(19 + i, 0, "| %-4d | %-15s | %-10s | %-5d | %-19s |",
                 i + 1, score_entries[i].name, score_entries[i].character_name, score_entries[i].score, score_entries[i].time_str);
    }
    mvprintw(19 + (score_count < 10 ? score_count : 10), 0, "+------+-----------------+------------+-------+---------------------+");

    refresh();
    getch(); // Wait for user input before exiting
}

void generate_obstacle(Position *obstacle, int min_x) {
    obstacle->x = min_x + MIN_OBSTACLE_DISTANCE + (rand() % MIN_OBSTACLE_DISTANCE);
    obstacle->y = GROUND_Y - (rand() % (GROUND_Y - OBSTACLE_MAX_HEIGHT + 1));
    if (obstacle->y < OBSTACLE_MAX_HEIGHT) {
        obstacle->y = OBSTACLE_MAX_HEIGHT;
    }
    if (obstacle->y > OBSTACLE_MIN_HEIGHT) {
        obstacle->y = OBSTACLE_MIN_HEIGHT;
    }
}

int main() {
    initscr();
    init_colors();
    noecho();
    curs_set(FALSE);
    timeout(-1); // Blocking input for name and character selection
    keypad(stdscr, TRUE);
    srand(time(NULL));

    Player player;
    Character selected_character;
    int scores[100]; // Store scores of each try
    int try_count = 0;

    // Get player name and select character
    get_player_name(&player);
    selected_character = select_character(player.character_name);

    // Read the high score from scores.txt
    int high_score = read_high_score();

    timeout(0); // Non-blocking input for game play

    Position dino = {5, GROUND_Y};
    Position obstacles[MAX_OBSTACLES];
    Position clouds[MAX_CLOUDS];
    Position item = {-1, -1}; // Initialize item position to off-screen
    int item_type = 0; // No item initially
    int score = 0;
    int game_over = 0;
    int initial_run = 1;
    int jumping = 0;
    int jump_counter = 0;
    int delay = INITIAL_DELAY;
    int max_obstacles = INITIAL_OBSTACLES;
    int last_speed_increase = 0;
    int last_obstacle_increase = 0;
    int crouching = 0;
    int crouch_counter = 0;
    int item_active = 0;
    int reverse_controls = 0;
    int theme = 0; // 0 = day, 1 = night
    int score_saved = 0; // Track if the score has been saved
    int paused = 0; // Paused state

    reset_game(&dino, obstacles, clouds, &score, &game_over, &initial_run, &jumping, &jump_counter, &delay, &max_obstacles, &last_speed_increase, &last_obstacle_increase, &crouching, &crouch_counter, &item, &item_active, &reverse_controls, &theme);

    while (1) {
        if (!paused) {
            clear();

            // Draw clouds or stars based on theme
            if (!game_over) {
                for (int i = 0; i < MAX_CLOUDS; i++) {
                    if (theme == 0) {
                        draw_cloud(clouds[i]);
                    } else {
                        draw_star(clouds[i]);
                    }
                    clouds[i].x--;
                    if (clouds[i].x < 0) {
                        clouds[i].x = COLS;
                        clouds[i].y = rand() % (GROUND_Y / 2);
                    }
                }
            } else {
                for (int i = 0; i < MAX_CLOUDS; i++) {
                    if (theme == 0) {
                        draw_cloud(clouds[i]);
                    } else {
                        draw_star(clouds[i]);
                    }
                }
            }

            // Draw sun or moon based on theme
            if (theme == 0) {
                draw_sun();
            } else {
                draw_moon();
            }

            // Draw ground
            for (int x = 0; x < COLS; x++) {
                mvprintw(GROUND_Y + DINO_HEIGHT, x, "-");
            }

            if (!game_over) {
                // Handle jumping
                if (jumping) {
                    if (jump_counter < JUMP_DELAY) {
                        dino.y--;
                    } else if (jump_counter < 2 * JUMP_DELAY) {
                        dino.y++;
                    } else {
                        jumping = 0;
                        jump_counter = 0;
                        dino.y = GROUND_Y; // Ensure the dino lands exactly on the ground
                    }
                    jump_counter++;
                }

                // Handle crouching
                if (crouching) {
                    if (crouch_counter < CROUCH_DURATION) {
                        crouch_counter++;
                    } else {
                        crouching = 0;
                        crouch_counter = 0;
                    }
                }

                // Draw dino
                draw_dino(dino, crouching, selected_character);

                // Draw and move obstacles
                for (int i = 0; i < max_obstacles; i++) {
                    draw_obstacle(obstacles[i]);
                    obstacles[i].x--;
                    if (obstacles[i].x + OBSTACLE_WIDTH < 0) {
                        generate_obstacle(&obstacles[i], obstacles[(i - 1 + max_obstacles) % max_obstacles].x);
                        if (obstacles[i].x < COLS) {
                            obstacles[i].x = COLS;
                        }
                        if (!initial_run) {
                            score += 50; // Increase score by 50 when an obstacle is passed
                        }
                    }

                    // Check collision
                    if (check_collision(dino, obstacles[i], crouching)) {
                        game_over = 1;
                        player.score = score;
                        player.is_restart = 0; // Not a restart
                        player.time = time(NULL); // Record the time of game over
                        if (!score_saved) { // Ensure score is saved only once
                            save_score(player);
                            scores[try_count++] = score; // Save the score of this try
                            score_saved = 1;
                        }
                    }
                }

                // Handle item
                if (!item_active && (rand() % 500) == 0) { // Randomly generate an item
                    item_active = 1;
                    item.x = COLS;
                    item.y = GROUND_Y - 5; // Adjust item height to be reachable by jumping
                    item_type = (rand() % 3) + 1; // Random item type
                }

                if (item_active) {
                    draw_item(item, item_type);
                    item.x--;

                    if (check_item_collision(dino, item)) {
                        apply_item_effect(item_type, &delay, &reverse_controls, &theme);
                        item_active = 0; // Remove item after collision
                        item.x = -1; // Hide item
                        item.y = -1; // Hide item
                    } else if (item.x < 0) {
                        item_active = 0; // Remove item if it goes off screen
                    }
                }

                if (obstacles[0].x + OBSTACLE_WIDTH == COLS - 1) {
                    initial_run = 0; // Initial run completed
                }

                // Increase difficulty with score
                if (score >= last_speed_increase + SPEED_INCREASE_INTERVAL) {
                    delay -= SPEED_INCREASE_AMOUNT;
                    if (delay < MIN_DELAY) {
                        delay = MIN_DELAY; // Set a minimum delay to prevent the game from becoming too fast
                    }
                    last_speed_increase = score; // Update last speed increase score
                }

                // Increase number of obstacles with score
                if (score >= last_obstacle_increase + OBSTACLE_INCREASE_INTERVAL) {
                    if (max_obstacles < MAX_OBSTACLES) {
                        // Add one new obstacle at the right edge of the screen
                        obstacles[max_obstacles].x = COLS + MIN_OBSTACLE_DISTANCE;
                        obstacles[max_obstacles].y = GROUND_Y - (rand() % (GROUND_Y - OBSTACLE_MAX_HEIGHT + 1));
                        max_obstacles++;
                    }
                    last_obstacle_increase = score; // Update last obstacle increase score
                }

                // Display score and high score
                mvprintw(0, 0, "Score: %d  High Score: %d", score, high_score);
            } else {
                // Game over message
                mvprintw(LINES / 2, COLS / 2 - 5, "Game Over!");
                mvprintw(LINES / 2 + 1, COLS / 2 - 10, "Press 'r' to try again.");
                mvprintw(LINES / 2 + 2, COLS / 2 - 10, "Press 'q' to exit the game.");
            }

            // Handle input
            int ch = getch();
            if (ch == 'q') {
                player.score = score;
                player.is_restart = 0; // Final score save, not a restart
                player.time = time(NULL); // Record the time of game exit
                if (!score_saved) { // Ensure score is saved only once
                    save_score(player);
                }
                break;
            }

            if ((!reverse_controls && (ch == ' ' || ch == KEY_UP)) || (reverse_controls && ch == KEY_DOWN)) {
                if (!jumping && !game_over) { // Space bar or Up arrow key for jumping
                    jumping = 1;
                    crouching = 0;
                    crouch_counter = 0;
                }
            }

            if ((!reverse_controls && ch == KEY_DOWN) || (reverse_controls && (ch == ' ' || ch == KEY_UP))) {
                if (!jumping && !game_over) { // Down arrow key for crouching
                    crouching = 1;
                    crouch_counter = 0;
                }
            }

            if (ch == 'r' && game_over) { // 'r' key to restart
                player.score = score;
                player.is_restart = 1; // It is a restart
                player.time = time(NULL); // Record the time of restart
                reset_game(&dino, obstacles, clouds, &score, &game_over, &initial_run, &jumping, &jump_counter, &delay, &max_obstacles, &last_speed_increase, &last_obstacle_increase, &crouching, &crouch_counter, &item, &item_active, &reverse_controls, &theme);
                score_saved = 0; // Reset score saved flag for the new game
            }

            if (ch == 'p' && !game_over) { // 'p' key to pause
                paused = 1;
                clear();
                mvprintw(LINES / 2, COLS / 2 - 5, "Game Paused");
                mvprintw(LINES / 2 + 1, COLS / 2 - 10, "Press 'p' to resume.");
                refresh();
            }

            refresh();
            if (!game_over) {
                usleep(delay);
                score++; // Increase score by 1 every frame only if the game is not over
            }
        } else {
            int ch = getch();
            if (ch == 'p') {
                paused = 0;
            }
        }
    }

    endwin();

    print_game_over(player, scores, try_count);

    return 0;
}

